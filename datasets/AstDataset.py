from torch.utils.data import IterableDataset
import csv
import torch
import json
from treelstm import calculate_evaluation_orders
from utils.TreeLstmUtils import calculate_evaluation_orders_topdown
import os
import bz2
import math


class AstDataset(IterableDataset):
    "AST trees dataset"

    def __init__(self, data_path, vocabulary, max_tree_size=-1, nr_of_names_to_keep=300, remove_non_res=False, get_statistics_only=False):
        self.max_tree_size = max_tree_size
        self.vocabulary = vocabulary
        self.remove_non_res = remove_non_res
        self.get_statistics_only = get_statistics_only
        self.nr_of_names_to_keep = nr_of_names_to_keep

        if os.path.isfile(data_path):
            self.file_paths = [data_path]
        else:
            self.file_paths = []
            for dirpath, _, files in os.walk(data_path):
                self.file_paths += [os.path.join(dirpath, file)
                                    for file in files]

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

        if num_shareable_files == 0:
            unshared_files = self.file_paths
            shared_files = []
        else:
            unshared_files = self.file_paths[:-num_shareable_files]
            shared_files = self.file_paths[-num_shareable_files:]
            workers_per_shared_file = num_workers / len(shared_files)

        # first iterate over files that can be read independently
        for file_index, file_path in enumerate(unshared_files):
            if file_index % num_workers == worker_id:
                # Create CSV file iterator
                if file_path.endswith('.bz2'):
                    file_iterator = Bz2CsvLineReader(file_path).read_csv()
                else:
                    file_iterator = csv.reader(open(file_path))

                # To skip the header, call next so the iterator starts from the first line
                next(file_iterator)

                for ast in file_iterator:
                    if self.get_statistics_only:
                        nodes, depths, ast_id = self.get_statistics(ast)
                        if (self.max_tree_size == -1 or nodes <= self.max_tree_size) and nodes > 1 and depths > 0:
                            yield nodes, depths, ast_id
                    else:
                        tree, nodes, ast_id = self.preprocess(ast)
                        if (self.max_tree_size == -1 or nodes <= self.max_tree_size) and nodes > 1 and len(tree) > 0:
                            yield tree, ast_id

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
                        if self.get_statistics_only:
                            nodes, depths, ast_id = self.get_statistics(ast)
                            if (self.max_tree_size == -1 or nodes <= self.max_tree_size) and nodes > 1 and depths > 0:
                                yield nodes, depths, ast_id
                        else:
                            tree, nodes, ast_id = self.preprocess(ast)
                            if (self.max_tree_size == -1 or nodes <= self.max_tree_size) and nodes > 1 and len(tree) > 0:
                                yield tree, ast_id

    def preprocess(self, ast):
        # load the JSON of the tree
        tree = json.loads(ast[1])

        # Get the amount of nodes
        nodes = self.get_amt_nodes(tree)

        if nodes > 1:
            try:
                tree = self.convert_tree_to_tensors(tree)
            except Exception as e:
                tree = {}
        else:
            tree = {}

        return tree, nodes, ast[0]

    def get_statistics(self, ast):
        # load the JSON of the tree
        tree = json.loads(ast[1])

        # Get the amount of nodes
        nodes = self.get_amt_nodes(tree)
        depths = self.get_max_depth(tree)

        return nodes, depths, ast[0]

    def get_amt_nodes(self, root, nodes=0):
        if 'children' in root:
            to_remove = []
            for child in root['children']:
                if not child['res'] and 'children' in child and not self.remove_non_res:
                    to_remove.append(child)
                if child['res'] or not self.remove_non_res:
                    nodes += self.get_amt_nodes(child, 0)
                else:
                    to_remove.append(child)

            for child in to_remove:
                root['children'].remove(child)

        return nodes + 1

    def get_max_depth(self, root):
        if 'children' in root:
            return 1 + max(self.get_max_depth(child) for child in root['children'])
        else:
            return 1

    def _label_node_index(self, node, n=0):
        node['index'] = n
        if 'children' in node:
            for child in node['children']:
                n += 1
                n = self._label_node_index(child, n)
        return n

    def _gather_node_attributes(self, node, key, parent=None, declared_names=None):
        if declared_names is None:
            declared_names = DeclaredNames(self.nr_of_names_to_keep)

        if self.vocabulary.token2index is not None:
            if node['res']:
                feature = self.vocabulary.token2index['RES'][node[key]]
                vocab = 'RES'
            else:
                if 'LITERAL' in self.vocabulary.token2index.keys() and 'LITERAL' in parent[key]:
                    feature = self.vocabulary.token2index['LITERAL'][node[key]]
                    vocab = 'LITERAL'
                elif 'TYPE' in self.vocabulary.token2index.keys() and 'TYPE' == parent[key]:
                    feature = self.vocabulary.token2index['TYPE'][node[key]]
                    vocab = 'TYPE'
                elif 'NAME_BUILTIN' in self.vocabulary.token2index.keys() and 'REF_BUILTIN' in parent[key]:
                    feature = self.vocabulary.token2index['NAME_BUILTIN'][node[key]]
                    vocab = 'NAME_BUILTIN'
                elif 'NAME' in self.vocabulary.token2index.keys():
                    token = node[key]

                    if 'decl_line' in node:
                        decl_line = node['decl_line']
                    else:
                        decl_line = None

                    if token in list(self.vocabulary.token2index['NAME'].keys())[:self.nr_of_names_to_keep]:
                        feature = self.vocabulary.token2index['NAME'][token]

                    elif not 'REF' in parent[key] or not declared_names.is_declared(token, decl_line):
                        feature = declared_names.add_feature(token, decl_line)

                    else:
                        feature = declared_names.get_feature(token, decl_line)

                    #     feature = len(oov_name_token2index) + self.nr_of_names_to_keep + self.vocabulary.offsets['NAME']
                    #     oov_name_token2index[token + '_' + str(token['index'])] = feature

                    # else:
                    #     feature = oov_name_token2index[token + '_' + str(token['index'])]



                    # elif token in oov_name_token2index:
                    #     feature = oov_name_token2index[token]

                    # else:
                    #     feature = len(oov_name_token2index) + self.nr_of_names_to_keep + self.vocabulary.offsets['NAME']
                    #     oov_name_token2index[token] = feature

                    # name_id = self.vocabulary.token2index['NAME'][node[key]]
                    # if name_id < self.nr_of_names_to_keep:
                    #     feature = name_id

                    # Map the name token to an ID -> if already mapped, get the ID, if not: add to mapping
                    # elif name_id in nameid_to_placeholderid:
                    #     feature = nameid_to_placeholderid[name_id]
                    # else:
                    #     feature = len(nameid_to_placeholderid) + self.nr_of_names_to_keep
                    #     nameid_to_placeholderid[name_id] = feature

                    vocab = 'NAME'

                    # if not 'REF' in parent[key]:

                else:
                    feature = self.vocabulary.token2index['NON_RES'][node[key]]
                    vocab = 'NON_RES'


            if vocab == 'NAME' and declared_names.is_declared(node[key]):
                feature_combined = feature
            else:
                feature_combined = self.vocabulary.token2index['ALL'][node[key]]

            # if 'DECL' in node[key] and 'DECLARATOR' != node[key] and 'DECL_STMT' != parent[key]:
                # names = self._find_names_decl(node, key)
                # if node[key] not in ['DECL_STMT', 'FUNCTION_DECL', 'VAR_DECL']:
                # print(parent[key], node['index'], names)
                # self._find_names_decl(node, key)


        else:
            feature = node[key]
            vocab = ''

        features = [[torch.tensor([feature])]]
        features_combined = [[torch.tensor([feature_combined])]]
        vocabs = [vocab]

        if 'children' in node:
            for idx, child in enumerate(node['children']):
                feature, feature_combined, vocab, declared_names = self._gather_node_attributes(
                    child, key, node, declared_names)
                features.extend(feature)
                features_combined.extend(feature_combined)
                vocabs += vocab

        return features, features_combined, vocabs, declared_names


    def _find_names_decl(self, node, key):
        names = []

        if 'children' in node:
            for child in node['children']:
                if not child['res']\
                   and not 'LITERAL' in node[key]\
                   and 'TYPE' != node[key]\
                   and 'REF_BUILTIN' != node[key]\
                   and 'PARM_DECL' != child[key]\
                   and 'REF' not in node[key]:
                    names.append(child[key])
                    return names

                if 'COMPOUND_STMT' != child[key]:
                    names.extend(self._find_names_decl(child, key))


        return names




    def _gather_adjacency_list(self, node):
        adjacency_list = []
        if 'children' in node:
            for child in node['children']:
                adjacency_list.append([node['index'], child['index']])
                adjacency_list.extend(self._gather_adjacency_list(child))

        return adjacency_list

    def _gather_adjacency_list_sib(self, node):
        adjacency_list = []
        if 'children' in node:
            for child_idx in range(len(node['children'])):
                if child_idx > 0:
                    adjacency_list.append([node['children'][child_idx - 1]['index'], node['children'][child_idx]['index']])
                else:
                    adjacency_list.append([0, node['children'][child_idx]['index']])
                adjacency_list.extend(self._gather_adjacency_list_sib(node['children'][child_idx]))

        return adjacency_list

    def convert_tree_to_tensors(self, tree):
        # Label each node with its walk order to match nodes to feature tensor indexes
        # This modifies the original tree as a side effect
        amt_nodes = self._label_node_index(tree)

        declared_names = [[] for _ in range(amt_nodes)]

        features, features_combined, vocabs, declared_names = self._gather_node_attributes(
            tree, 'token', declared_names)
        adjacency_list = self._gather_adjacency_list(tree)
        adjacency_list_sib = self._gather_adjacency_list_sib(tree)

        node_order_bottomup, edge_order_bottomup = calculate_evaluation_orders(
            adjacency_list, len(features))
        node_order_topdown, edge_order_topdown, edge_order_topdown_sib = calculate_evaluation_orders_topdown(
            adjacency_list, adjacency_list_sib, len(features))

        return {
            'features': torch.tensor(features, dtype=torch.int32),
            'features_combined': torch.tensor(features_combined, dtype=torch.int32),
            'node_order_bottomup': torch.tensor(node_order_bottomup, dtype=torch.int32),
            'node_order_topdown': torch.tensor(node_order_topdown, dtype=torch.int32),
            'adjacency_list': torch.tensor(adjacency_list, dtype=torch.int64),
            'adjacency_list_sib': torch.tensor(adjacency_list_sib, dtype=torch.int64),
            'edge_order_bottomup': torch.tensor(edge_order_bottomup, dtype=torch.int32),
            'edge_order_topdown': torch.tensor(edge_order_topdown, dtype=torch.int32),
            'edge_order_topdown_sib': torch.tensor(edge_order_topdown_sib, dtype=torch.int32),
            'vocabs': vocabs,
            'declared_names': declared_names
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


class DeclaredNames():
    def __init__(self, idx_offset):
        self.names = {}
        self.idx_offset = idx_offset

    def add_feature(self, name, line=None):
        if self.is_declared(name, line):
            return self.get_feature(name, line)
        else:
            if line:
                key = f'{name}_{line}'
            else:
                key = name

            self.names[key] = len(self.names) + self.idx_offset

            return self.names[key]

    def get_feature(self, name, line=None):
        if line:
            key = f'{name}_{line}'
            if key in self.names:
                return self.names[key]
            else:
                return self.names[name]

        else:
            name_idx = ['_'.join(k.split('_')[:-1]) for k in self.names.keys()].index(name)
            key = list(self.names.keys())[name_idx]

            return self.names[key]

    def is_declared(self, name, line=None):
        if line:
            return f'{name}_{line}' in self.names.keys()
        else:
            return name in ['_'.join(k.split('_')[:-1]) for k in self.names.keys()]

    def get_name(self, feature):
        features = list(self.names.values())

        if feature in features:
            return '_'.join(list(self.names.keys())[features.index(feature)].split('_')[:-1])
        else:
            return -1


    def get_closest_feature(self, name, line):
        names_tokens = ['_'.join(k.split('_')[:-1]) for k in self.names.keys()]
        lines = [int(k.split('_')[-1]) for k in self.names.keys()]

        closest_distance = math.inf
        feature = -1

        for idx, (name_tok, name_line) in enumerate(zip(names_tokens, lines)):
            if name_tok == name and math.fabs(name_line - line) < closest_distance:
                closest_distance = math.fabs(name_line - line)
                feature = list(self.names.values())[idx]

        return feature





