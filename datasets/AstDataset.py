from torch.utils.data import IterableDataset
import csv
import torch
import json
from treelstm import calculate_evaluation_orders
from TreeLstmUtils import calculate_evaluation_orders_topdown


class AstDataset(IterableDataset):
    "AST trees dataset"
    
    def __init__(self, csv_path, vocab_size, max_tree_size=-1):
        self.csv_path = csv_path
        self.vocab_size = vocab_size
        self.max_tree_size = max_tree_size
    
    def __iter__(self):
        # Create CSV file iterator
        file_iterator = csv.reader(open(self.csv_path))
        # To skip the header, call next so the iterator starts from the first line
        next(file_iterator)
        
        # Get worker info, id and number of workers
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        
        # Iterate over the CSV file
        for i, ast in enumerate(file_iterator):
            # Since each worker iterates over the same CSV file
            # To avoid duplicate data, each worker reads unique lines from the dataset
            # e.g. 4 workers -> worker 0 reads lines 0, 4, 8 and worker 1 reads lines 1, 5, 9 etc...
            # This is not optimal but certainly a speed up compared to just using 0 workers
            if i % num_workers == worker_id:
                tree, nodes = self.preprocess(ast)
                if (self.max_tree_size == -1 or nodes <= self.max_tree_size) and nodes > 1:
                    yield tree      
        
    def preprocess(self, ast):
        # load the JSON of the tree
        tree = json.loads(ast[1])
        # Remove the non-reserved keyword nodes
        nodes = self.remove_non_res_nodes(tree)
        
        if nodes > 1:
            tree = self.convert_tree_to_tensors(tree)
        else:
            tree = {}
            
        return tree, nodes
        
    def remove_non_res_nodes(self, root, nodes=0):
        if 'children' in root:
            to_remove = []
            for child in root['children']:
                if child['res']: 
                    nodes += self.remove_non_res_nodes(child)
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


    def _gather_node_attributes(self, node, key):
        features = [[torch.tensor([node[key]])]]
        if 'children' in node:
            for child in node['children']:
                features.extend(self._gather_node_attributes(child, key))
        return features


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

        features = self._gather_node_attributes(tree, 'token')
        adjacency_list = self._gather_adjacency_list(tree)
        
        node_order_bottomup, edge_order_bottomup = calculate_evaluation_orders(adjacency_list, len(features))
        node_order_topdown, edge_order_topdown = calculate_evaluation_orders_topdown(adjacency_list, len(features))

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'node_order_bottomup': torch.tensor(node_order_bottomup, dtype=torch.int64),
            'node_order_topdown': torch.tensor(node_order_topdown, dtype=torch.int64),
            'adjacency_list': torch.tensor(adjacency_list, dtype=torch.int64),
            'edge_order_bottomup': torch.tensor(edge_order_bottomup, dtype=torch.int64),
            'edge_order_topdown': torch.tensor(edge_order_topdown, dtype=torch.int64),
        }               
    