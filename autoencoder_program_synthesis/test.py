from models.Vae import Vae
from utils.TreeLstmUtils import batch_tree_input
from datasets.AstDataset import AstDataset
from utils.ModelResults import ModelResults
import os
import csv
import torch
from torch.utils.data import DataLoader, BufferedShuffleDataset
import sys
from model_utils.vocabulary import Vocabulary
from models.TreeLstmEncoderComplete import TreeLstmEncoderComplete
from models.TreeLstmDecoderComplete import TreeLstmDecoderComplete
from model_utils.metrics_helper import MetricsHelperTree2Tree
from config.vae_config import ex
import pandas as pd
pd.options.mode.chained_assignment = None


torch.multiprocessing.set_sharing_strategy('file_system')
maxInt = sys.maxsize

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


@ex.config
def set_config():
    pretrained_model = 'checkpoints/1/model.tar'

    folder = os.path.dirname(pretrained_model)
    
    for file in os.listdir(folder):
        if 'config' in file and file.endswith('json'):
            ex.add_config(os.path.join(folder, file))


    # Overwrite config pretrained model
    ex.add_config({'pretrained_model': pretrained_model, 'batch_size': 1})


class Tester:
    def __init__(self):
        self.vocabulary = self.get_vocabulary()
        self.loss_weights = self.get_loss_weights()
        self.model = self.make_model()
        self.load_model()
        self.test_dataset = self.get_dataset()
        self.test_loader = self.get_dataloader()


    def make_model(self):
        model = Vae(device,
                    TreeLstmEncoderComplete, 
                    TreeLstmDecoderComplete, 
                    self.vocabulary, 
                    MetricsHelperTree2Tree, 
                    self.loss_weights
                    ).to(device)
                    
        return model


    @ex.capture
    def load_model(self, pretrained_model):
        self.model.load_model(pretrained_model)


    @ex.capture
    def get_vocabulary(self, tokens_paths, max_name_tokens, reusable_name_tokens):
        max_tokens = {
            'RES': None,
            'NAME': max_name_tokens + reusable_name_tokens,
            'TYPE': None,
            'LITERAL': None,
            'NAME_BUILTIN': None,
        }

        vocabulary = Vocabulary(tokens_paths, max_tokens)

        return vocabulary        

    @ex.capture
    def get_loss_weights(self, weighted_loss):
        loss_weights = {}

        for vocab_type, token_counts in self.vocabulary.token_counts.items():
            if weighted_loss:
                counts = torch.tensor(list(token_counts.values())[:self.vocabulary.get_vocab_size(vocab_type)])
                loss_weights[vocab_type] = (torch.min(counts, dim=-1)[0] / counts) * torch.max(counts, dim=-1)[0]
            else:
                loss_weights[vocab_type] = torch.ones(self.vocabulary.get_vocab_size(vocab_type))

        return loss_weights


    @ex.capture
    def get_dataset(self, dataset_paths, max_tree_size, max_name_tokens, batch_size):
        test_dataset = AstDataset(dataset_paths['TEST'], self.vocabulary, max_tree_size, max_name_tokens)
        test_dataset = BufferedShuffleDataset(test_dataset, buffer_size=batch_size)

        return test_dataset


    @ex.capture
    def get_dataloader(self, batch_size, num_workers):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            collate_fn=batch_tree_input,
            num_workers=num_workers)

        
        return test_loader


    @ex.capture
    def run(self, save_dir, _run, dataset_paths, temperature, top_k, top_p, latent_dim):
        if save_dir is not None:
            save_dir = os.path.join(save_dir, str(_run._id))
            os.makedirs(save_dir, exist_ok=True)

        test_programs = pd.read_csv(dataset_paths['TEST_PROGRAMS'])

        bleu_scores, perc_compiles = self.model.test(self.test_loader, temperature, top_k, top_p, save_folder=str(_run._id), test_programs=test_programs)

        # self.model.generate(torch.randn([1000, latent_dim], device=device), str(_run._id), 0.7, 40, 0.9)
        
        # self.model.interpolate(self.test_loader, 5, str(_run._id), temperature, top_k, top_p)

        bleu_scores = 0
        perc_compiles = 0   

        return bleu_scores, perc_compiles



@ex.main
def main(pretrained_model):
    # pretrained model should be given
    assert pretrained_model is not None and os.path.isfile(pretrained_model)

    tester = Tester()
    bleu_scores, perc_compiles = tester.run()

    return {'bleu_scores': bleu_scores, 'percentage compiles': perc_compiles}

if __name__ == "__main__":
    ex.run_commandline()
