import json
import csv
import torch
from torch.utils.data import DataLoader
from torch import optim
import sys
sys.path.append("utils/")
from datasets.AstDataset import AstDataset
from utils.TreeLstmUtils import batch_tree_input
from models.TreeLstmEncoder import TreeLstmEncoder
from models.TreeLstmDecoder import TreeLstmDecoder
from models.Vae import Vae
from loss_functions.TreeVaeLoss import TreeVaeLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv.field_size_limit(sys.maxsize)

# HYPERPARAMETERS
EMBEDDING_DIM = 30
HIDDEN_SIZE = 512
LATENT_DIM = 256
LEARNING_RATE = 0.005
EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 8


def train(dataset_path, reserved_tokens_path):
    # Load the reserved tokens dictionary
    with open(reserved_tokens_path, 'r') as json_f:
        json_data = json_f.read()

    # To JSON format (dictionary)
    reserved_tokens = json.loads(json_data)
    vocab_size = len(reserved_tokens)

    ast_dataset = AstDataset(dataset_path, vocab_size=vocab_size, max_tree_size=-1)

    loader = DataLoader(ast_dataset, batch_size=BATCH_SIZE, collate_fn=batch_tree_input, num_workers=NUM_WORKERS)
    
    
    # Load models
    encoder = TreeLstmEncoder(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, LATENT_DIM, device)
    decoder = TreeLstmDecoder(vocab_size, HIDDEN_SIZE, LATENT_DIM, device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    vae_loss = TreeVaeLoss()
    
    vae = Vae(encoder, decoder, encoder_optimizer, decoder_optimizer, vae_loss, vocab_size, device)
    
    # Train
    vae.train(loader, EPOCHS, save_dir='checkpoints/')
    

if __name__ == "__main__":
    reserved_tokens_path = '../data/ast_trees_200k/reserved_tokens.json'
    dataset_path = '../data/ast_trees_200k/asts.csv'
    train(dataset_path, reserved_tokens_path)