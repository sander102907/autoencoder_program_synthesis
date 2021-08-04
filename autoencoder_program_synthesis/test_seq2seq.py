from models.SeqRnnEncoder import SeqRnnEncoder
from models.Vae import Vae
from datasets.SeqDataset import SeqDataset
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import json
from model_utils.vocabulary import Vocabulary
from config.vae_config import ex
from models.SeqRnnEncoder import SeqRnnEncoder
from models.SeqRnnDecoder import SeqRnnDecoder
from model_utils.metrics_helper import MetricsHelperSeq2Seq


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@ex.config
def set_config():
    pretrained_model = '../vastai/10&9_800latent_seq2seq/model.tar' # '../vastai/4&8_500latent_seq2seq/model.tar' 
    folder = os.path.dirname(pretrained_model)
    
    for file in os.listdir(folder):
        if 'config' in file and file.endswith('json'):
            ex.add_config(os.path.join(folder, file))
            break
            
    dataset_paths = {
        # 'TRAIN': '../data/ast_trees_full_19-06-2021/asts_train/',
        # 'VAL': '../data/ast_trees_full_19-06-2021/asts_val/',
        # 'TEST': '../data/ast_trees_full_19-06-2021/asts_test/',
        # 'TEST_PROGRAMS': '../data/ast_trees_full_19-06-2021/programs_test.csv'


        # For the seq2seq model
        'TRAIN': '../data/seq_data/programs_train/',
        'VAL': '../data/seq_data/programs_val/',
        'TEST': '../data/seq_data/programs_test/',
        'TEST_SMALL': '../data/seq_data/programs_test_small/',
        'TEST_RECON': '../data/seq_data/programs_test_temp/'
    } 


    # Overwrite config pretrained model
    ex.add_config({'pretrained_model': pretrained_model, 'batch_size': 5, 'dataset_paths': dataset_paths})


class Tester:
    def __init__(self):
        self.vocabulary = self.get_vocabulary()
        self.loss_weights = self.get_loss_weights()
        self.model = self.make_model()
        self.load_model()
        self.test_dataset = self.get_datasets()
        self.test_loader = self.get_dataloaders()



    @ex.capture
    def make_model(self):
        model = Vae(device, 
                    SeqRnnEncoder, 
                    SeqRnnDecoder, 
                    self.vocabulary, 
                    MetricsHelperSeq2Seq, 
                    self.loss_weights
                    ).to(device)

        return model


    @ex.capture
    def load_model(self, pretrained_model):
        if pretrained_model is not None and os.path.isfile(pretrained_model):
            self.model.load_model(pretrained_model)


    @ex.capture
    def get_vocabulary(self, tokens_paths):
        max_tokens = {
            'ALL': None
        }

        tokens_paths = {'ALL': tokens_paths['ALL']}

        vocabulary = Vocabulary(tokens_paths, max_tokens, add_special_tokens=True)

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
        test_files = [os.path.join(dataset_paths['TEST_RECON'], file) for file in os.listdir(dataset_paths['TEST_RECON'])]

        test_datasets = list(map(lambda x : SeqDataset(x, self.vocabulary, max_program_size, device), test_files))
        test_dataset = ConcatDataset(test_datasets)

        return test_dataset


    @ex.capture
    def get_dataloaders(self, batch_size, num_workers):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers)

        
        return test_loader



    @ex.capture
    def save_config(self, save_dir, _config):
        config_path = os.path.join(save_dir, 'config.json')

        with open(config_path, 'w') as f:
            json.dump(_config, f)



    @ex.capture
    def run(self, save_dir, _run, temperature, top_k, top_p, latent_dim):
        if save_dir is not None:
            save_dir = os.path.join(save_dir, str(_run._id))
            os.makedirs(save_dir, exist_ok=True)
            self.save_config(save_dir)
            

        # bleu_scores, perc_compiles = self.model.test(self.test_loader, temperature, top_k, top_p, save_folder=str(_run._id))  
        self.model.generate(torch.randn([1000, latent_dim], device=device), str(_run._id), 0.7, 40, 0.9)
        
        # self.model.interpolate(self.test_loader, 5, str(_run._id), temperature, top_k, top_p)

        bleu_scores = 0
        perc_compiles = 0
 

        return bleu_scores, perc_compiles


@ex.main
def main(pretrained_model):
    # pretrained model should be given
    assert pretrained_model is not None and os.path.isfile(pretrained_model)

    print(set_config())

    tester = Tester()
    bleu_scores, perc_compiles = tester.run()

    return {'bleu scores': bleu_scores, 'percentage compiles': perc_compiles}

if __name__ == "__main__":
    ex.run_commandline()
