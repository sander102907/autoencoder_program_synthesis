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
from sacred import Experiment
from sacred.observers import MongoObserver
from config.vae_config import ex


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
    pretrained_model = 'checkpoints/546/iter30.tar'

    folder = os.path.dirname(pretrained_model)
    
    for file in os.listdir(folder):
        if 'config' in file and file.endswith('json'):
            ex.add_config(os.path.join(folder, file))


    # Overwrite config pretrained model
    ex.add_config({'pretrained_model': 'checkpoints/546/iter30.tar'})


class Tester:
    def __init__(self):
        self.vocabulary = self.get_vocabulary()
        self.loss_weights = self.get_loss_weights()
        self.model = self.make_model()
        self.load_model()
        self.test_dataset = self.get_dataset()
        self.test_loader = self.get_dataloader()


    def make_model(self):
        model = Vae(device, self.vocabulary, self.loss_weights).to(device)
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
    def run(self, save_dir, _run):
        if save_dir is not None:
            save_dir = os.path.join(save_dir, str(_run._id))
            os.makedirs(save_dir, exist_ok=True)

        avg_tree_bleu_scores, seq_bleu_scores = self.model.test(self.test_loader, str(_run._id))

        return avg_tree_bleu_scores, seq_bleu_scores



@ex.main
def main(pretrained_model):
    # pretrained model should be given
    assert pretrained_model is not None and os.path.isfile(pretrained_model)

    tester = Tester()
    avg_tree_bleu_scores, seq_bleu_scores = tester.run()
    
    # results = ModelResults()
    # results.from_dict(bleu_scores)

    return {'seq_bleu_scores': seq_bleu_scores, 'avg_tree_bleu_scores': avg_tree_bleu_scores}

if __name__ == "__main__":
    ex.run_commandline()