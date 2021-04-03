import json
import csv
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("utils/")
from datasets.AstDataset import AstDataset
from utils.TreeLstmUtils import batch_tree_input
from models.Vae import Vae
from loss_functions.TreeVaeLoss import TreeVaeLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv.field_size_limit(sys.maxsize)

# HYPERPARAMETERS
params = {
    'EMBEDDING_DIM': 30,
    'HIDDEN_SIZE': 512,
    'LATENT_DIM': 512,
    'LEARNING_RATE': 0.005,
    'EPOCHS': 10,
    'BATCH_SIZE': 32,
    'NUM_WORKERS': 8,
    'CLIP': 5,
    'KL_LOSS_WEIGHT': 10,
}


def train(dataset_path, reserved_tokens_path):
    # Load the reserved tokens dictionary
    with open(reserved_tokens_path, 'r') as json_f:
        json_data = json_f.read()

    # To JSON format (dictionary)
    reserved_tokens = json.loads(json_data)
    label_to_idx = {k:i for i, k in enumerate(reserved_tokens.keys())}
    vocab_size = len(reserved_tokens)
    params['VOCAB_SIZE'] = vocab_size

    ast_dataset = AstDataset(dataset_path, vocab_size=vocab_size, label_to_idx=label_to_idx, max_tree_size=300)

    loader = DataLoader(ast_dataset, batch_size=params['BATCH_SIZE'], collate_fn=batch_tree_input, num_workers=params['NUM_WORKERS'])
    
    
    # Load loss
    vae_loss = TreeVaeLoss(params['KL_LOSS_WEIGHT'])
    
    # set model
    vae = Vae(device, params, vae_loss)
    
    # Train
    vae.train(loader, params['EPOCHS'], save_dir='checkpoints/')
    

if __name__ == "__main__":
    #reserved_tokens_path = '../data/ast_trees_200k/reserved_tokens.json'
    #dataset_path = '../data/ast_trees_200k/asts.csv'
    reserved_tokens_path = '../data/ast_trees_solutions600/reserved_tokens_0.json'
    dataset_path = '../data/ast_trees_solutions600/asts/asts0.csv.bz2'
    train(dataset_path, reserved_tokens_path)
