import json
import os
import re
import math

class Vocabulary:
    def __init__(self, paths, max_tokens, add_special_tokens=False):
        self.token_counts = {}
        self.token2index = {}
        self.index2token = {}
        # self.offsets = {}


        self.max_tokens = max_tokens

        for name, path in paths.items():
            self.load_token_counts(path, name, add_special_tokens)

        # self.token_counts['ALL'] = {k:v for k,v in self.token_counts['ALL'].items() if v > 50}

        self.create_tokens()

        if not 'ALL' in paths.keys():
            self.create_combined_vocabs()        
        

    def load_token_counts(self, path, name, add_special_tokens):
        self.token_counts[name] = {}
        
        # self.offsets[name] = 0

        # add offsets for token vocabs to be able to create a combined vocab
        # for k in self.token_counts.keys():
        #     if self.max_tokens[k] is not None:
        #         self.offsets[name] += min(len(self.token_counts[k]), self.max_tokens[k])
        #     else:
        #         self.offsets[name] += len(self.token_counts[k])

        if add_special_tokens:
            self.token_counts[name]['<pad>'] = math.inf
            self.token_counts[name]['<unk>'] = math.inf
            self.token_counts[name]['<sos>'] = math.inf
            self.token_counts[name]['<eos>'] = math.inf


        # If path is a file
        if os.path.isfile(path):
            self.load_token_counts_file(path, name)
        # If path is a folder
        else:
            self.load_token_counts_folder(path, name)

        # sort tokens by frequence of occurence, most occuring first
        self.token_counts[name] = dict(sorted(self.token_counts[name].items(), key=lambda x:x[1], reverse=True)[:self.max_tokens[name]])
            

    def load_token_counts_file(self, path, name):
         # Load the reserved tokens dictionary
        with open(path, 'r') as json_f:
            json_data = json_f.read()

        # To JSON format (dictionary)
        self.token_counts[name] = json.loads(json_data)


    def load_token_counts_folder(self, path, name):
        for dirpath, _, files in os.walk(path):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(dirpath, file), 'r') as json_f:
                        json_data = json_f.read()

                    # To JSON format (dictionary)
                    for k, v in json.loads(json_data).items():
                        if k in self.token_counts[name]:
                            self.token_counts[name][k] += v
                        else:
                            self.token_counts[name][k] = v


    def create_tokens(self):
        for name, token_count in self.token_counts.items():
            self.token2index[name] = {k: i for i, k in enumerate(token_count.keys())}
            self.index2token[name] = {v: k for k, v in self.token2index[name].items()}


    def create_combined_vocabs(self):
        self.token2index['ALL'] = {}
        self.index2token['ALL'] = {}

        for vocab in self.token2index.keys():
            if vocab != 'ALL':
                for token in self.token2index[vocab].keys():
                    if token not in self.token2index['ALL']:
                        self.token2index['ALL'][token] = len(self.token2index['ALL'])

        # for name, offset in self.offsets.items():
        #     for token, index in self.token2index[name].items():
        #         self.token2index['ALL'][token] = index + offset
                
                
        self.index2token['ALL'] = {v: k for k, v in self.token2index['ALL'].items()}

    def get_vocab_size(self, name):
        return len(self.token2index[name])


    def add_new_token(self, name, token, index=None):
        if index is None:
            index = len(self.token2index[name])
            self.token2index[name][token] = index
            self.index2token[name][index] = token

        if name != 'ALL':
            index_all = len(self.token2index['ALL'])
            self.token2index['ALL'][token] = index_all
            self.index2token['ALL'][index_all] = token

    def get_tokens(self, name):
        return list(self.token_counts[name].keys())


    def get_cleaned_index2token(self, name, max=None):
        abbr_to_full = {
            'ref': ['reference'],
            'decl': ['declaration'],
            'stmt': ['statement'],
            'expr': ['expression'],
            'bool': ['boolean'],
            'cxx': ['c++'],
            'def': ['definition'],
            'attr': ['attribute'],
            'var': ['variable'],
            'const': ['constant'],
            'parm': ['parameter'],
            'bitor': ['bit', 'or'],
            'lvaluereference': ['left', 'value', 'reference'],
            'rvaluereference': ['right', 'value', 'reference'],
            'goto': ['go', 'to'],
            'addr': ['address'],
            'bitand': ['bit', 'and'],
            'stmtexpr': ['statement', 'expression'],
            'functionproto': ['function', 'prototype'],
            'typedef': ['type', 'definition'],
            'typeid': ['type', 'identifier'],
            'paren': ['parent'],
            'sizeof': ['size', 'of'],
            'cstyle': ['c', 'style'],
            }

        tokens = list(self.index2token[name].values())
        cleaned_index2token = {}

        for token in tokens[:max]:
            # To lowercase and split on '_' and ' ' 
            split_tokens = re.split('_| ', token.lower())

            cleaned_token = []
           
            for split_token in split_tokens:
                if split_token in abbr_to_full:
                    cleaned_token.extend(abbr_to_full[split_token])  
                elif 'operator' in split_token and not 'operator' == split_token:
                    cleaned_token.extend(['operator', split_token.split('operator')[-1]])
                else:
                    cleaned_token.append(split_token)
                    
            cleaned_index2token[len(cleaned_index2token)] = cleaned_token
        
        return cleaned_index2token


    