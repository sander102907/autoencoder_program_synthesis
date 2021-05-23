from loss_functions.TreeVaeLoss import TreeVaeLoss, TreeVaeLossComplete
from models.Vae import Vae
from utils.TreeLstmUtils import batch_tree_input
from datasets.AstDataset import AstDataset
import os
import json
import csv
import torch
from torch.utils.data import DataLoader, BufferedShuffleDataset
import sys
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


# HYPERPARAMETERS
params = {
    'LEAF_EMBEDDING_DIM': 100,
    'EMBEDDING_DIM': 50,
    'HIDDEN_SIZE': 48,
    'LATENT_DIM': 48,
    'LEARNING_RATE': 1e-3,
    'NUM_LSTM_LAYERS': 2,
    'EPOCHS': 30,
    'BATCH_SIZE': 64,
    'NUM_WORKERS': 8,
    'CLIP_GRAD_NORM': 0,            # clip the gradient norm, setting to 0 ignores this
    'CLIP_GRAD_VAL': 0,             # clip the gradient value, setting to 0 ignores this
    'KL_LOSS_WEIGHT': 0.001,
    'USE_CELL_LSTM_OUTPUT': False,
    'VAE': False,
    # Whether to weight the loss: with imbalanced vocabularies to how often the tokens occur
    'WEIGHTED_LOSS': False,
    # Whether to use individual LSTM layers for each of the different vocabularies
    'INDIV_LAYERS_VOCABS': False,
    # TODO: implement teacher forcing ratio -> does it make sense, e.g. predicted func decl but should be var decl then how does it work with children?
    'TEACHER_FORCING_RATIO': 0.5,
    'TOP_NAMES_TO_KEEP': 300,
    # vocabulary size for name tokens which are mapped to non-unique IDs should be high enough to cover all programs
    'NAME_ID_VOCAB_SIZE': 100,
    'SAVE_PER_BATCHES': 1000
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
                    for k, v in json.loads(json_data).items():
                        if k in tokens:
                            tokens[k] += v
                        else:
                            tokens[k] = v

    return tokens


def train(dataset_path_train, dataset_path_val, tokens_paths=None, tokenized=False):
    token_vocabs = {}
    label_to_idx = None
    idx_to_label = None

    for k, path in tokens_paths.items():
        token_vocabs[k] = load_token_vocabulary(path)
        params[f'{k}_VOCAB_SIZE'] = len(token_vocabs[k])

        if params['WEIGHTED_LOSS']:
            loss_weights = 1 / torch.tensor(list(token_vocabs[k].values()))
            params[f'{k}_WEIGHTS'] = loss_weights / \
                torch.sum(loss_weights) * len(token_vocabs[k]) * 1000
        else:
            if k == 'NAME':
                params[f'{k}_WEIGHTS'] = torch.ones(
                    params['TOP_NAMES_TO_KEEP'] + params['NAME_ID_VOCAB_SIZE'])
            else:
                params[f'{k}_WEIGHTS'] = torch.ones(len(token_vocabs[k]))

    os.makedirs('output/', exist_ok=True)

    if not tokenized:
        label_to_idx = {}
        idx_to_label = {}
        for k, vocab in token_vocabs.items():
            vocab = dict(sorted(vocab.items(), key=lambda x:x[1], reverse=True))
            label_to_idx[k] = {k: i for i, k in enumerate(vocab.keys())}
            idx_to_label[k] = {v: k for k, v in label_to_idx[k].items()}

            # Save to file to be used when transforming back to code
            json_f = json.dumps(label_to_idx[k])
            f = open(f'output/{k}_tokens.json', 'w')
            f.write(json_f)
            f.close()

    non_res_tokens = len(tokens_paths) > 1

    self.params['TOKEN_VOCABS'] = token_vocabs

    # weights_res = 1 / torch.tensor(list(token_vocabs['RES'].values()))
    # params['WEIGHTS_RES'] = weights_res / torch.sum(weights_res)

    train_dataset = AstDataset(dataset_path_train, label_to_idx,
                               max_tree_size=750, remove_non_res=not non_res_tokens,
                               nr_of_names_to_keep=params['TOP_NAMES_TO_KEEP'])

    train_dataset = BufferedShuffleDataset(train_dataset, buffer_size=128)

    val_dataset = AstDataset(dataset_path_val, label_to_idx,
                             max_tree_size=750, remove_non_res=not non_res_tokens,
                             nr_of_names_to_keep=params['TOP_NAMES_TO_KEEP'])

    val_dataset = BufferedShuffleDataset(val_dataset, buffer_size=128)

    train_loader = DataLoader(
        train_dataset, batch_size=params['BATCH_SIZE'], collate_fn=batch_tree_input, num_workers=params['NUM_WORKERS'])
    val_loader = DataLoader(
        val_dataset, batch_size=params['BATCH_SIZE'], collate_fn=batch_tree_input, num_workers=params['NUM_WORKERS'])

    # set model
    vae = Vae(device, params)

    save_dir = 'checkpoints/' + f'{params["LATENT_DIM"]}latent' + f'_{params["HIDDEN_SIZE"]}hidden' + \
        f'{"_weightedloss" if params["WEIGHTED_LOSS"] else ""}' + \
        f'{"_indivlayers" if params["INDIV_LAYERS_VOCABS"] else ""}' + '/'

    os.makedirs(save_dir, exist_ok=True)

    # Train
    vae.train(params['EPOCHS'], train_loader, val_loader, save_dir)


if __name__ == "__main__":
    tokens_paths = {
        'RES': '../data/ast_trees_full_22-04-2021/reserved_tokens/',
        'NAME': '../data/ast_trees_full_22-04-2021/name_tokens/',
        'TYPE': '../data/ast_trees_full_22-04-2021/type_tokens/',
        'LITERAL': '../data/ast_trees_full_22-04-2021/literal_tokens/',
    }
    dataset_path_train = '../data/ast_trees_full_22-04-2021/asts_train/'
    dataset_path_val = '../data/ast_trees_full_22-04-2021/asts_val/'

    train(dataset_path_train, dataset_path_val, tokens_paths)
