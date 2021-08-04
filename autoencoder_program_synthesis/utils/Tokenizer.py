import os
import json

class Tokenizer:
    def __init__(self, out_path):
        self.out_path = out_path
        self.token_dict = {}
        self.label_dict = {}

    def get_token(self, label):
        if label in self.token_dict:
            return self.token_dict[label]
        else:
            token = len(self.token_dict)
            self.token_dict[label] = token
            self.label_dict[token] = label
            return token

    def get_label(self, token):
        if token in self.label_dict:
            return self.label_dict[token]
        else:
            return ''

    def save(self):
        if not self.out_path:
            return

        json_f = json.dumps(self.token_dict)
        f = open(self.out_path, 'w')
        f.write(json_f)
        f.close()

    def load(self, path):
        with open(path, 'r') as json_f:
            json_data = json_f.read()

        self.token_dict = json.loads(json_data)

        # Reverse the token dict to create a label dict
        self.label_dict = dict((reversed(item) for item in self.token_dict.items()))
