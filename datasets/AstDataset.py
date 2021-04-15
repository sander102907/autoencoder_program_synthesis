from torch.utils.data import IterableDataset
import csv
import torch
import json
from treelstm import calculate_evaluation_orders
from TreeLstmUtils import calculate_evaluation_orders_topdown
import os
import bz2
from functools import partial
import math

class AstDataset(IterableDataset):
    "AST trees dataset"
    
    def __init__(self, data_path, label_to_idx, max_tree_size=-1, remove_non_res=False):
        self.max_tree_size = max_tree_size
        self.label_to_idx = label_to_idx
        self.remove_non_res = remove_non_res
        
        if os.path.isfile(data_path):
            self.file_paths = [data_path]
        else:
            self.file_paths = []
            for dirpath, _, files in os.walk(data_path):
                self.file_paths += [os.path.join(dirpath, file) for file in files]
            
    def __iter__(self):        
        # Get worker info, id and number of workers
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        
        # Calculate files which can be read by the workers independently
        # So each 36 files with 8 workers, each worker can read at most 4 files independently
        # Because 36 / 8 = 4,5, then the last 4 files need to be shared by 8 workers
        # So each worker will read half a file in this example
        num_shareable_files = len(self.file_paths) % num_workers
        unshared_files = self.file_paths[:-num_shareable_files]
        shared_files = self.file_paths[-num_shareable_files:]
        workers_per_shared_file = num_workers / len(shared_files)   
        
        # first iterate over files that can be read independently
        for file_index, file_path in enumerate(unshared_files):
            if file_index % num_workers == worker_id:
                print(f'{worker_id}: {file_path}\n')
                # Create CSV file iterator
                if file_path.endswith('.bz2'):
                    file_iterator = Bz2CsvLineReader(file_path).read_csv()
                else:
                    file_iterator = csv.reader(open(file_path))

                # To skip the header, call next so the iterator starts from the first line
                next(file_iterator)
                
                for ast in file_iterator:
                    tree, nodes = self.preprocess(ast)
                    if (self.max_tree_size == -1 or nodes <= self.max_tree_size) and nodes > 1:
                        yield tree 
        
        # iterate over files that have to be shared between workers
        for file_index, file_path in enumerate(shared_files):
            if math.floor(worker_id / workers_per_shared_file) == file_index:
                # Create CSV file iterator
                if file_path.endswith('.bz2'):
                    file_iterator = Bz2CsvLineReader(file_path).read_csv()
                else:
                    file_iterator = csv.reader(open(file_path))
                # To skip the header, call next so the iterator starts from the first line
                next(file_iterator)
                
                for i, ast in enumerate(file_iterator):
                    # Since multiple workers iterate over the same CSV file
                    # To avoid duplicate data, each worker reads unique lines from the dataset
                    # e.g. 4 workers -> worker 0 reads lines 0, 4, 8 and worker 1 reads lines 1, 5, 9 etc...
                    # This is not optimal but certainly a speed up compared to just using 1 worker for each file
                    if worker_id - (file_index * workers_per_shared_file) == i % workers_per_shared_file:
                        tree, nodes = self.preprocess(ast)
                        if (self.max_tree_size == -1 or nodes <= self.max_tree_size) and nodes > 1:
                            yield tree 
                    
        
    def preprocess(self, ast):
        # load the JSON of the tree
        tree = json.loads(ast[1])

        # Get the amount of nodes
        nodes = self.get_amt_nodes(tree)
        
        if nodes > 1:
            tree = self.convert_tree_to_tensors(tree)
        else:
            tree = {}
            
        return tree, nodes
        
    def get_amt_nodes(self, root, nodes=0):
        if 'children' in root:
            to_remove = []
            for child in root['children']:
                if not child['res'] and 'children' in child and not self.remove_non_res:
                    to_remove.append(child)
                if child['res'] or not self.remove_non_res:
                    nodes += self.get_amt_nodes(child)
                else:
                    to_remove.append(child)

            for child in to_remove:
                root['children'].remove(child)

        return nodes + 1
    
                    
    def _label_node_index(self, node, n=0):
        node['index'] = n
        if 'children' in node:
            for child in node['children']:
                n += 1
                n = self._label_node_index(child, n)
        return n


    def _gather_node_attributes(self, node, key, parent=None):
        if self.label_to_idx is not None:
            if node['res']:
                feature = self.label_to_idx['RES'][node[key]]
                vocab = 'RES'
            else:
                if 'LITERAL' in self.label_to_idx.keys() and 'LITERAL' in parent['token']:
                    feature = self.label_to_idx['LITERAL'][node[key]]
                    vocab = 'LITERAL'
                elif 'TYPE' in self.label_to_idx.keys() and 'TYPE' == parent['token']:
                    feature = self.label_to_idx['TYPE'][node[key]]
                    vocab = 'TYPE'
                elif 'NAME' in self.label_to_idx.keys():
                    feature = self.label_to_idx['NAME'][node[key]]
                    vocab = 'NAME'
                else:
                    feature = self.label_to_idx['NON_RES'][node[key]]
                    vocab = 'NON_RES'
        else:
            feature = node[key]      
            vocab = ''          

        features = [[torch.tensor([feature])]]
        vocabs = [vocab]
        if 'children' in node:
            for child in node['children']:
                feature, vocab = self._gather_node_attributes(child, key, node) 
                features.extend(feature)
                vocabs += vocab
                # vocabs.extend(vocab)

        # print(vocabs)
        return features, vocabs


    def _gather_adjacency_list(self, node):
        adjacency_list = []
        if 'children' in node:
            for child in node['children']:
                adjacency_list.append([node['index'], child['index']])
                adjacency_list.extend(self._gather_adjacency_list(child))

        return adjacency_list

    def convert_tree_to_tensors(self, tree):
        # Label each node with its walk order to match nodes to feature tensor indexes
        # This modifies the original tree as a side effect
        self._label_node_index(tree)

        features, vocabs = self._gather_node_attributes(tree, 'token')
        adjacency_list = self._gather_adjacency_list(tree)

        
        node_order_bottomup, edge_order_bottomup = calculate_evaluation_orders(adjacency_list, len(features))
        node_order_topdown, edge_order_topdown = calculate_evaluation_orders_topdown(adjacency_list, len(features))

        return {
            'features': torch.tensor(features, dtype=torch.int32),
            'node_order_bottomup': torch.tensor(node_order_bottomup, dtype=torch.int32),
            'node_order_topdown': torch.tensor(node_order_topdown, dtype=torch.int32),
            'adjacency_list': torch.tensor(adjacency_list, dtype=torch.int64),
            'edge_order_bottomup': torch.tensor(edge_order_bottomup, dtype=torch.int32),
            'edge_order_topdown': torch.tensor(edge_order_topdown, dtype=torch.int32),
            'vocabs': vocabs
        }
    
    
    
class Bz2CsvLineReader():
    """Line reader to read bz2 compressed CSV file line by line, helpful for the iterable dataset"""
    def __init__(self, filename):
        self.filename = filename
        
    def _line_reader(self, file):
        for line in file:
            line = line.decode('utf-8')
            yield line
        
    def read_csv(self):
        with bz2.BZ2File(self.filename, "r") as file:
            for i, row in enumerate(csv.reader(self._line_reader(file))):
                yield row

#     def readlines(self):
#         with open(self.filename, 'rb') as file:
#             for row in csv.reader(self._line_reader(file)):
#                 yield row

#     def _line_reader(self, file):
#         buffer = ''
#         decompressor = bz2.BZ2Decompressor()
#         reader = partial(file.read, self.buffer_size)

#         for bindata in iter(reader, b''):
#             block = decompressor.decompress(bindata).decode('utf-8')
#             buffer += block
#             if '\n' in buffer:
#                 lines = buffer.splitlines(True)
#                 if lines:
#                     buffer = '' if lines[-1].endswith('\n') else lines.pop()
#                     for line in lines:
#                         yield line
    