from models.SeqRnnEncoder import SeqRnnEncoder
from torch.utils import data
from models.Vae import Vae
from datasets.SeqDataset import SeqDataset
import utils.KLScheduling as KLScheduling
import os
import torch
import math
from torch.utils.data import DataLoader, ConcatDataset
import json
from model_utils.vocabulary import Vocabulary
from torch import optim
from config.vae_config import ex
from random import sample
from models.SeqRnnEncoder import SeqRnnEncoder
from models.SeqRnnDecoder import SeqRnnDecoder
from model_utils.metrics_helper import MetricsHelperSeq2Seq


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        self.vocabulary = self.get_vocabulary()
        self.add_special_vocab_tokens()
        self.loss_weights = self.get_loss_weights()
        print('making model')
        self.model = self.make_model()
        self.set_optimizer()
        self.load_model()
        print('making datasets')
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()
        print('getting scheduler')
        self.kl_scheduler = self.get_kl_scheduler()



    @ex.capture
    def make_model(self):
        model = Vae(device, SeqRnnEncoder, SeqRnnDecoder, self.vocabulary, MetricsHelperSeq2Seq, self.loss_weights).to(device)
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
    def get_vocabulary(self, tokens_paths):
        max_tokens = {
            'ALL': None
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
    def get_datasets(self, dataset_paths, max_program_size):
        # files = set(os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if 'programs' in file)

        # train_files, val_files, test_files = self.create_train_val_test_split(files)

        train_files = [os.path.join(dataset_paths['TRAIN'], file) for file in os.listdir(dataset_paths['TRAIN'])]
        val_files = [os.path.join(dataset_paths['VAL'], file) for file in os.listdir(dataset_paths['VAL'])]
        test_files = [os.path.join(dataset_paths['TEST'], file) for file in os.listdir(dataset_paths['TEST'])]

        train_datasets = list(map(lambda x : SeqDataset(x, self.vocabulary, max_program_size, device), train_files))
        train_dataset = ConcatDataset(train_datasets)

        val_datasets = list(map(lambda x : SeqDataset(x, self.vocabulary, max_program_size, device), val_files))
        val_dataset = ConcatDataset(val_datasets)

        test_datasets = list(map(lambda x : SeqDataset(x, self.vocabulary, max_program_size, device), test_files))
        test_dataset = ConcatDataset(test_datasets)

        self.num_train_programs = sum([len(dset) for dset in train_datasets])

        return train_dataset, val_dataset, test_dataset


    def create_train_val_test_split(self, files, val_test_files=13):
        test_files = sample(files, val_test_files)
        
        files = files - set(test_files)

        val_files = sample(files, val_test_files)

        train_files = files - set(val_files)

        return train_files, val_files, test_files


    @ex.capture
    def get_dataloaders(self, batch_size, num_workers):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers)

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers)

        
        return train_loader, val_loader, test_loader


    @ex.capture
    def get_kl_scheduler(self, kl_scheduling, kl_warmup_iters, kl_weight, kl_ratio, kl_function, kl_cycles, batch_size, num_epochs):
        assert kl_scheduling in ['constant', 'monotonic', 'cyclical']

        if kl_scheduling == 'constant':
            return KLScheduling.ConstantAnnealing(kl_warmup_iters, kl_weight)

        elif kl_scheduling == 'monotonic':
            iterations = math.ceil((self.num_train_programs / batch_size) * num_epochs)
            return KLScheduling.MonotonicAnnealing(iterations, kl_warmup_iters, kl_ratio, kl_function)

        elif kl_scheduling == 'cyclical':
            iterations = math.ceil((self.num_train_programs / batch_size) * num_epochs)
            return KLScheduling.CyclicalAnnealing(iterations, kl_warmup_iters, kl_cycles, kl_ratio, kl_function)


    def add_special_vocab_tokens(self):
        self.vocabulary.add_new_token('ALL', '<pad>')
        self.vocabulary.add_new_token('ALL', '<unk>')
        self.vocabulary.add_new_token('ALL', '<sos>')
        self.vocabulary.add_new_token('ALL', '<eos>')




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
        bleu_scores, perc_compiles = self.model.test(self.test_loader, save_folder=str(_run._id))       

        return bleu_scores, perc_compiles


@ex.main
def main(_run):
    trainer = Trainer()
    bleu_scores, perc_compiles = trainer.run()

    return {'bleu scores': bleu_scores, 'percentage compiles': perc_compiles}

if __name__ == "__main__":
    ex.run_commandline()
