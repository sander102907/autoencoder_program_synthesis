from autoencoder_program_synthesis.models.Vae import Vae
from autoencoder_program_synthesis.models.TreeLstmDecoderComplete import TreeLstmDecoderComplete
from autoencoder_program_synthesis.models.TreeLstmEncoderComplete import TreeLstmEncoderComplete
from autoencoder_program_synthesis.model_utils.vocabulary import Vocabulary
from autoencoder_program_synthesis.datasets.AstDataset import AstDataset
from os import path
import os
import json
from cpp_ast_parser.AST_parser import AstParser
import torch
from anytree.exporter import JsonExporter

class Tree2Tree():
    def __init__(self,
                 libclang_path: str,
                 checkpoint_folder: os.path):

        assert os.path.isdir(checkpoint_folder), f"Make sure checkpoint_folder: {checkpoint_folder} is a directory containing 'model.tar', 'config.json' and the tokens subfolders."

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_folder = checkpoint_folder

        self.ast_parser = AstParser(libclang_path)

        tokens_paths = {
            'NAME':         os.path.join(checkpoint_folder, 'name_tokens/'),
            'NAME_BUILTIN': os.path.join(checkpoint_folder, 'name_builtin_tokens/'),
            'RES':          os.path.join(checkpoint_folder, 'reserved_tokens/'),
            'TYPE':         os.path.join(checkpoint_folder, 'type_tokens/'),
            'LITERAL':      os.path.join(checkpoint_folder, 'literal_tokens/'),
        }

        self.vocabulary = self.__get_vocabulary(tokens_paths)
        self.loss_weights = self.__get_loss_weights(self.vocabulary, weighted_loss=False)

        self.dataset = AstDataset('', self.vocabulary)
        self.ast_exporter = JsonExporter()

        self.model = Vae(self.device,
            TreeLstmEncoderComplete, 
            TreeLstmDecoderComplete, 
            self.vocabulary,
            None,
            self.loss_weights
            )

        self.__init_model()


    def __init_model(self):
        pretrained_model_path = os.path.join(self.checkpoint_folder, 'model.tar')
        config_path = os.path.join(self.checkpoint_folder, 'config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)


        self.model.create_embedding_layers(
            config['embedding_dim'], 
            config['indiv_embed_layers'], 
            config['pretrained_emb'], 
            pretrained_model_path)

        self.model.encoder = TreeLstmEncoderComplete(self.device, 
                                                self.model.embedding_layers, 
                                                config['embedding_dim'],
                                                config['rnn_hidden_size'],
                                                config['latent_dim'],
                                                config['use_cell_output_lstm'],
                                                config['vae'],
                                                config['num_rnn_layers_enc'],
                                                config['recurrent_dropout'],
                                                config['indiv_embed_layers'])


        self.model.decoder = TreeLstmDecoderComplete(self.device, 
                                                self.model.embedding_layers, 
                                                self.vocabulary, 
                                                self.loss_weights,
                                                config['embedding_dim'],
                                                config['rnn_hidden_size'],
                                                config['latent_dim'],
                                                config['use_cell_output_lstm'],
                                                config['num_rnn_layers_dec'],
                                                config['dropout'],
                                                config['recurrent_dropout'],
                                                config['indiv_embed_layers'],
                                                config['max_name_tokens'])

        self.model = self.model.to(self.device)
        
        self.model.load_model(pretrained_model_path)



    def __get_vocabulary(self, tokens_paths):
        max_tokens = {
            'RES': None,
            'NAME': 150,
            'TYPE': None,
            'LITERAL': None,
            'NAME_BUILTIN': None,
        }

        vocabulary = Vocabulary(tokens_paths, max_tokens)

        return vocabulary  

    def __get_loss_weights(self, vocabulary, weighted_loss):
            loss_weights = {}

            for vocab_type, token_counts in vocabulary.token_counts.items():
                if weighted_loss:
                    counts = torch.tensor(list(token_counts.values())[:vocabulary.get_vocab_size(vocab_type)])
                    loss_weights[vocab_type] = (torch.min(counts, dim=-1)[0] / counts) * torch.max(counts, dim=-1)[0]
                else:
                    loss_weights[vocab_type] = torch.ones(vocabulary.get_vocab_size(vocab_type))

            return loss_weights


    def encode(self, program: str) -> torch.Tensor:
        ast = self.ast_parser.string_to_ast(program)
        ast_json = json.loads(self.ast_exporter.export(ast))

        model_inp, amt_nodes = self.dataset.process_ast(ast_json)
        model_inp['tree_sizes'] = [amt_nodes]

        for key in model_inp.keys():
            if key not in ['ids', 'tree_sizes', 'vocabs', 'declared_names', 'length', 'id']:
                model_inp[key] = model_inp[key].to(self.device)

        z = self.model.encode(model_inp)

        return z


    def decode(self, z: torch.Tensor, temperature: float, top_k: int, top_p: float) -> list:
        program = self.model.decode(z, temperature, top_k, top_p)

        return program

