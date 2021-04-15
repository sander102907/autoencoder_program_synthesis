import os
import json
import csv
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("utils/")
from datasets.AstDataset import AstDataset
from utils.TreeLstmUtils import batch_tree_input
from models.Vae import Vae
from loss_functions.TreeVaeLoss import TreeVaeLoss, TreeVaeLossComplete
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv.field_size_limit(sys.maxsize)

# HYPERPARAMETERS
params = {
    'LEAF_EMBEDDING_DIM': 100,
    'EMBEDDING_DIM': 30,
    'HIDDEN_SIZE': 512,
    'LATENT_DIM': 512,
    'LEARNING_RATE': 0.005,
    'EPOCHS': 10,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 8,
    'CLIP': 5,
    'KL_LOSS_WEIGHT': 0.001,
}

def load_token_vocabulary(path):
    if os.path.isfile(path):
        # Load the reserved tokens dictionary
        with open(path, 'r') as json_f:
            json_data = json_f.read()

        # To JSON format (dictionary)
        tokens = json.loads(json_data)

    else:
        tokens = {}

        for dirpath, _, files in os.walk(path):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(dirpath, file), 'r') as json_f:
                        json_data = json_f.read()

                    # To JSON format (dictionary)
                    for k,v in json.loads(json_data).items():
                            if k in tokens:
                                tokens[k] += v
                            else:
                                tokens[k] = v

    return tokens


def train(dataset_path, tokens_paths=None, tokenized=False):
    token_vocabs = {}
    label_to_idx = None
    
    for k, path in tokens_paths.items():
        token_vocabs[k] = load_token_vocabulary(path)
        params[f'{k}_VOCAB_SIZE'] = len(token_vocabs[k])
        
    if len(tokens_paths) > 1:
        # Load loss
        vae_loss = TreeVaeLossComplete(params['KL_LOSS_WEIGHT'])
    else:       
        vae_loss = TreeVaeLoss(params['KL_LOSS_WEIGHT'])
            
    
    if not tokenized:
        label_to_idx = {}
        for k, vocab in token_vocabs.items():
            label_to_idx[k] = {k:i for i, k in enumerate(vocab.keys())}
            
    non_res_tokens = len(tokens_paths) > 1
    
    ast_dataset = AstDataset(dataset_path, label_to_idx, max_tree_size=200, remove_non_res=not non_res_tokens)

    loader = DataLoader(ast_dataset, batch_size=params['BATCH_SIZE'], collate_fn=batch_tree_input, num_workers=params['NUM_WORKERS'])
    
    # set model
    vae = Vae(device, params, vae_loss, non_res_tokens)
        
    # Train
    vae.train(loader, params['EPOCHS'], save_dir='checkpoints/')
    

if __name__ == "__main__":
    tokens_paths = {
        'RES': '../data/ast_trees/reserved_tokens.json',
        'NAME': '../data/ast_trees/name_tokens.json',
        'TYPE': '../data/ast_trees/type_tokens.json',
        'LITERAL': '../data/ast_trees/literal_tokens.json',
    }
    dataset_path = '../data/ast_trees/asts.csv.bz2'

    train(dataset_path, tokens_paths)
