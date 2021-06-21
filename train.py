from models.Vae import Vae
from utils.TreeLstmUtils import batch_tree_input
from datasets.AstDataset import AstDataset
import utils.KLScheduling as KLScheduling
from utils.ModelResults import ModelResults
import os
import csv
import torch
import math
from torch.utils.data import DataLoader, BufferedShuffleDataset
import sys
import json
from model_utils.vocabulary import Vocabulary
from torch import optim
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


class Trainer:
    def __init__(self):
        self.vocabulary = self.get_vocabulary()
        self.loss_weights = self.get_loss_weights()
        self.model = self.make_model()
        self.set_optimizer()
        self.load_model()
        self.kl_scheduler = self.get_kl_scheduler()
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()


    @ex.capture
    def make_model(self):
        model = Vae(device, self.vocabulary, self.loss_weights).to(device)
        return model

    @ex.capture
    def set_optimizer(self, learning_rate):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.set_optimizer(optimizer)
        return optimizer

    @ex.capture
    def load_model(self, pretrained_model):
        if pretrained_model is not None and os.path.isfile(pretrained_model):
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
    def get_datasets(self, dataset_paths, max_tree_size, max_name_tokens, batch_size):
        train_dataset = AstDataset(dataset_paths['TRAIN'], self.vocabulary, max_tree_size, max_name_tokens)
        train_dataset = BufferedShuffleDataset(train_dataset, buffer_size=batch_size)

        val_dataset = AstDataset(dataset_paths['VAL'], self.vocabulary, max_tree_size, max_name_tokens)
        val_dataset = BufferedShuffleDataset(val_dataset, buffer_size=batch_size)

        test_dataset = AstDataset(dataset_paths['TEST'], self.vocabulary, max_tree_size, max_name_tokens)
        test_dataset = BufferedShuffleDataset(test_dataset, buffer_size=batch_size)

        return train_dataset, val_dataset, test_dataset


    @ex.capture
    def get_dataloaders(self, batch_size, num_workers):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=batch_tree_input,
            num_workers=num_workers)

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            collate_fn=batch_tree_input,
            num_workers=num_workers)

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            collate_fn=batch_tree_input,
            num_workers=num_workers)

        
        return train_loader, val_loader, test_loader


    @ex.capture
    def get_kl_scheduler(self, kl_scheduling, kl_warmup_iters, kl_weight, kl_ratio, kl_function, kl_cycles, batch_size, num_epochs):
        assert kl_scheduling in ['constant', 'monotonic', 'cyclical']

        if kl_scheduling == 'constant':
            return KLScheduling.ConstantAnnealing(kl_warmup_iters, kl_weight)

        elif kl_scheduling == 'monotonic':
            iterations = math.ceil((self.vocabulary.token_counts['RES']['root'] / batch_size) * num_epochs)
            return KLScheduling.MonotonicAnnealing(iterations, kl_warmup_iters, kl_ratio, kl_function)

        elif kl_scheduling == 'cyclical':
            iterations = math.ceil((self.vocabulary.token_counts['RES']['root'] / batch_size) * num_epochs)
            return KLScheduling.CyclicalAnnealing(iterations, kl_warmup_iters, kl_cycles, kl_ratio, kl_function)



    @ex.capture
    def save_config(self, save_dir, _config):
        config_path = os.path.join(save_dir, 'config.json')

        with open(config_path, 'w') as f:
            json.dump(_config, f)



    @ex.capture
    def run(self, num_epochs, save_dir, _run):
        if save_dir is not None:
            save_dir = os.path.join(save_dir, str(_run._id))
            os.makedirs(save_dir, exist_ok=True)
            self.save_config(save_dir)
            

        self.model.fit(num_epochs, self.kl_scheduler, self.train_loader, self.val_loader, save_dir)
        avg_tree_bleu_scores, seq_bleu_scores = self.model.test(self.test_loader, str(_run._id))       

        return avg_tree_bleu_scores, seq_bleu_scores


@ex.main
def main(_run):
    trainer = Trainer()
    avg_tree_bleu_scores, seq_bleu_scores = trainer.run()
    
    # results = ModelResults()
    # results.from_dict(bleu_scores)

    return {'seq_bleu_scores': seq_bleu_scores, 'avg_tree_bleu_scores': avg_tree_bleu_scores}

if __name__ == "__main__":
    ex.run_commandline()
