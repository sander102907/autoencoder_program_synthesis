import os
from datetime import datetime
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
from models.TreeLstmEncoder import TreeLstmEncoder
from models.TreeLstmDecoder import TreeLstmDecoder
from models.TreeLstmEncoderComplete import TreeLstmEncoderComplete
from models.TreeLstmDecoderComplete import TreeLstmDecoderComplete
from time import time

class Vae():
    """
    A Vae model: wrapper for encoder, decoder models with train, evaluate and generate functions
    Can also be used to easily save and load models
    """
    
    def __init__(self, device, params, loss_function, non_res_tokens=True):
        self.res_vocab_size = params['RES_VOCAB_SIZE']
        self.clip = params['CLIP']
        self.non_res_tokens = non_res_tokens
        self.params = params

        self.embedding_layers = nn.ModuleDict({})

        # Create shared embedding layers based on vocab sizes we give
        for k, v in params.items():
            if 'VOCAB_SIZE' in k:
                # For reserved tokens
                if 'RES' in k:
                    embbedding_size = params['EMBEDDING_DIM']
                # For leaf tokens
                else:
                    embbedding_size = params['LEAF_EMBEDDING_DIM']

                self.embedding_layers[k.split('_')[0]] = nn.Embedding(params[k], embbedding_size)


        # If we are using leaf tokens as well, se the complete encoder and decoder models
        if len(self.embedding_layers) > 1:
            self.encoder = TreeLstmEncoderComplete(device, params, self.embedding_layers).to(device)
            self.decoder = TreeLstmDecoderComplete(device, params, self.embedding_layers).to(device)
        else:
            self.vocab_size = None
            self.embedding = None

            self.encoder = TreeLstmEncoder(device, params, self.res_embedding).to(device)
            self.decoder = TreeLstmDecoder(device, params, self.res_embedding).to(device)

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=params['LEARNING_RATE'])
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=params['LEARNING_RATE'])
        
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        
        # loss function
        self.loss_function = loss_function
        
        # Store losses -> so we can easily save them
        self.losses = {}
        
        self.device = device
        
    def train(self, data_loader, epochs, save_dir=None):
        """
        Trains the VAE model for the chosen encoder, decoder and its optimizers with the given loss function.
        @param data_loader: Torch Dataset that generates batches to train on
        @param epochs: The number of epochs to train for
        @param save_dir: The directory to save model checkpoints to, will save every epoch if save_path is given
        """
        
        self.encoder.train()
        self.decoder.train()
        
        save_loss_per_num_batches = 1
        loss_types = ['total_loss', 'sibling_loss', 'kl_loss']

        for k in self.embedding_layers.keys():
            loss_types.append(k + '_loss')
        
        running_losses = {}

        for loss_type in loss_types:
            self.losses[loss_type] = {}
        
        for epoch in range(epochs):
            pbar = tqdm(unit='batch', position=0)
            
            for loss_type in loss_types:
                running_losses[loss_type] = 0
            
            for batch_index, batch in enumerate(data_loader):
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                for key in batch.keys():
                    if key not in  ['tree_sizes', 'vocabs']:
                        batch[key] = batch[key].to(self.device)

                z, kl_loss = self.encoder(batch)
                reconstruction_loss = self.decoder(z, batch)
                loss = kl_loss * self.params['KL_LOSS_WEIGHT'] + reconstruction_loss
                loss.backward()

                # curr_losses = self.loss_function(output, z_mean, z_log_var)
                # curr_losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                pbar.set_postfix({'loss':loss.item(), 'kl_loss':kl_loss.item(), 'recon_loss':reconstruction_loss.item()})
                pbar.update()
                
                
                # for loss_type in loss_types:
                #     if loss_type == 'total_loss':
                #         running_losses[loss_type] += curr_losses[loss_type].item()
                #     else:
                #         running_losses[loss_type] += curr_losses[loss_type]
                
              
                # Add losses every "save_loss_per_num_batches"
                # if batch_index % save_loss_per_num_batches == save_loss_per_num_batches - 1:
                #     for loss_type in loss_types:
                #         self.losses[loss_type][f'epoch{epoch + 1}-batch{batch_index + 1}'] = running_losses[loss_type] / save_loss_per_num_batches 
                #         running_losses[loss_type] = 0
                    
#                 break
                    
                        
            if save_dir is not None:
                self.save_model(os.path.join(save_dir, f'VAE_epoch{epoch}_{datetime.now().strftime("%d-%m-%Y_%H:%M")}.tar'))
                        
        return self.losses
        
        
    def evaluate(self, batch):
        """
        Evaluates the VAE model: given data, reconstruct the input and output this
        @param data_loader: Torch Dataset that generates batches to evaluate on
        """
        
        self.encoder.eval()
        self.decoder.eval()
        
        reconstructions = []
        
        with torch.no_grad():
            for key in batch.keys():
                if key != 'tree_sizes':
                    batch[key] = batch[key].to(self.device)

            z, _, _ = self.encoder(batch)
            reconstructions += self.decoder(z)
                
        return reconstructions
            
        
        
    def generate(self, z):
        """
        Generate using the VAE model: given latent vector(s) z, generate output
        @param z: latent vector(s) -> (batch_size, latent_size)
        """
        
        self.decoder.eval()
        
        with torch.no_grad():
            output = self.decoder(z)
            
        return output
    
    
    def save_model(self, path):
        """
        Save the encoder, decoder and corresponding optimizer state dicts as well as losses to .tar
        Such that the model can be used to continue training or for inference
        @param path: Save the model to the given path
        """

        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        
        torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
                    'losses': self.losses
                    }, path)

    def load_model(self, path):
        """
        Load a model save to the encoder, decoder and corresponding optimizer state dicts and load losses
        Can be used to continue training a model or load a model for inference
        @param path: Load a model from the given path
        """
        
        checkpoint = torch.load(path)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        self.losses = checkpoint['losses']
