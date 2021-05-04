import os
from datetime import datetime
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import torch
from torch import optim
import torch.nn as nn
from models.TreeLstmEncoder import TreeLstmEncoder
from models.TreeLstmDecoder import TreeLstmDecoder
from models.TreeLstmEncoderComplete import TreeLstmEncoderComplete
from models.TreeLstmDecoderComplete import TreeLstmDecoderComplete
from time import time
from nltk.translate.bleu_score import corpus_bleu



class Vae():
    """
    A Vae model: wrapper for encoder, decoder models with train, evaluate and generate functions
    Can also be used to easily save and load models
    """

    def __init__(self, device, params):
        self.res_vocab_size = params['RES_VOCAB_SIZE']
        self.clip_grad_norm = params['CLIP_GRAD_NORM']
        self.clip_grad_val = params['CLIP_GRAD_VAL']
        self.params = params

        self.embedding_layers = nn.ModuleDict({})

        # Create shared embedding layers based on vocab sizes we give
        for k, v in params.items():
            if 'VOCAB_SIZE' in k:
                # For reserved tokens or if not individual layers for vocabs
                if 'RES' in k or not params['INDIV_LAYERS_VOCABS']:
                    embbedding_size = params['EMBEDDING_DIM']
                # For leaf tokens
                else:
                    embbedding_size = params['LEAF_EMBEDDING_DIM']

                if 'NAME' in k:
                    vocab_size = params['NAME_ID_VOCAB_SIZE']
                else:
                    vocab_size = params[k]

                self.embedding_layers[k.split('_')[0]] = nn.Embedding(
                    vocab_size, embbedding_size)

        # If we are using leaf tokens as well, se the complete encoder and decoder models
        if len(self.embedding_layers) > 1:
            self.encoder = TreeLstmEncoderComplete(
                device, params, self.embedding_layers).to(device)
            self.decoder = TreeLstmDecoderComplete(
                device, params, self.embedding_layers).to(device)
        else:
            self.vocab_size = None
            self.embedding = None

            self.encoder = TreeLstmEncoder(
                device, params, self.res_embedding).to(device)
            self.decoder = TreeLstmDecoder(
                device, params, self.res_embedding).to(device)

        encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=params['LEARNING_RATE'])
        decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=params['LEARNING_RATE'])

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

        # loss function
        # self.loss_function = loss_function

        # Store losses -> so we can easily save them
        self.losses = {}

        self.device = device

    def train(self, epochs, train_loader, val_loader=None, save_dir=None):
        """
        Trains the VAE model for the chosen encoder, decoder and its optimizers with the given loss function.
        @param epochs: The number of epochs to train for
        @param train_loader: Torch Dataset that generates batches to train on
        @param val_loader: Torch Dataset that generates batches to validate on
        @param save_dir: The directory to save model checkpoints to, will save every epoch if save_path is given
        """

        running_losses = {}
        loss_types_train = list(self.embedding_layers.keys()) + \
            ['PARENT', 'SIBLING', 'KL']

        loss_types_val = [f'VAL_{ltype}' for ltype in loss_types_train]

        for batch in train_loader:
            break


        for epoch in range(epochs):
            for loss_type in loss_types_train + loss_types_val:
                running_losses[loss_type] = 0
                self.losses[loss_type] = {}

            self.encoder.train()
            self.decoder.train()

            pbar = tqdm(unit='batch')
            for batch_index, batch in enumerate(train_loader):
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                for key in batch.keys():
                    if key not in ['tree_sizes', 'vocabs', 'nameid_to_placeholderid']:
                        batch[key] = batch[key].to(self.device)

                z, kl_loss = self.encoder(batch)
                reconstruction_loss, individual_losses, accuracies = self.decoder(
                    z, batch)

                # Calculate total loss and backprop
                loss = kl_loss * ((epoch/(epochs - 1)) +
                                    1e-8) + reconstruction_loss

                loss.backward()

                if self.clip_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.encoder.parameters(), self.clip_grad_norm)
                    torch.nn.utils.clip_grad_norm_(
                        self.decoder.parameters(), self.clip_grad_norm)

                if self.clip_grad_val != 0:
                    torch.nn.utils.clip_grad_value_(
                        self.encoder.parameters(), self.clip_grad_val)
                    torch.nn.utils.clip_grad_value_(
                        self.decoder.parameters(), self.clip_grad_val)

                # Update the weights
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                for loss_type in individual_losses.keys():
                    running_losses[loss_type] += individual_losses[loss_type]

                running_losses['KL'] += kl_loss.item()

                pbar.set_postfix({
                    'train_loss': loss.item(),
                    'kl_loss': kl_loss.item(),
                    'recon_loss': reconstruction_loss.item(),
                    'kl weight': (epoch/(epochs - 1)),
                    'acc_parent': accuracies['PARENT'],
                    'acc_sibling': accuracies['SIBLING'],
                    'acc_RES': accuracies['RES'],
                    'acc_NAME': accuracies['NAME'],
                    'acc_TYPE': accuracies['TYPE'],
                    'acc_LIT': accuracies['LITERAL']})
                pbar.update()

                if batch_index % self.params['SAVE_PER_BATCHES'] == self.params['SAVE_PER_BATCHES'] - 1:
                    for loss_type in loss_types_train:
                        self.losses[loss_type][f'epoch{epoch}-batch{batch_index}'] = running_losses[loss_type] / self.params['SAVE_PER_BATCHES']
                        running_losses[loss_type] = 0

                    if save_dir is not None:
                        self.save_model(os.path.join(
                            save_dir, f'VAE_epoch{epoch}_batch{batch_index}_{datetime.now().strftime("%d-%m-%Y_%H%M")}.tar'))


            if val_loader is not None:
                self.encoder.eval()
                self.decoder.eval()
                val_steps = 0

                for batch_index, batch in tqdm(enumerate(val_loader), unit='batch'):
                    with torch.no_grad():
                        for key in batch.keys():
                            if key not in ['tree_sizes', 'vocabs', 'nameid_to_placeholderid']:
                                batch[key] = batch[key].to(self.device)

                        z, kl_loss = self.encoder(batch)
                        reconstruction_loss, individual_losses, accuracies = self.decoder(
                            z, batch)

                        for loss_type in individual_losses.keys():
                            running_losses[f'VAL_{loss_type}'] += individual_losses[loss_type]

                        running_losses['VAL_KL'] += kl_loss.item()
                        val_steps += 1

                for loss_type in loss_types_val:
                    self.losses[loss_type][f'epoch{epoch}'] = running_losses[loss_type] / val_steps
                    running_losses[loss_type] = 0

            if save_dir is not None:
                self.save_model(os.path.join(
                    save_dir, f'VAE_epoch{epoch}_{datetime.now().strftime("%d-%m-%Y_%H%M")}.tar'))

        return self.losses

    def evaluate(self, batch, idx_to_label):
        """
        Evaluates the VAE model: given data, reconstruct the input and output this
        @param data_loader: Torch Dataset that generates batches to evaluate on
        """

        self.encoder.eval()
        self.decoder.eval()

        reconstructions = []

        with torch.no_grad():
            for key in batch.keys():
                if key not in ['tree_sizes', 'vocabs', 'nameid_to_placeholderid']:
                    batch[key] = batch[key].to(self.device)

            z, _ = self.encoder(batch)
            reconstructions += self.decoder(z, target=None, idx_to_label=idx_to_label,
                                            nameid_to_placeholderid=batch['nameid_to_placeholderid'])

        return reconstructions

    def generate(self, z, idx_to_label):
        """
        Generate using the VAE model: given latent vector(s) z, generate output
        @param z: latent vector(s) -> (batch_size, latent_size)
        """

        self.decoder.eval()

        with torch.no_grad():
            output = self.decoder(z, target=None, idx_to_label=idx_to_label)

        return output


    def calculate_eval_scores(self, trees, data, idx_to_label):
        """
        Calculate evaluation scores for generated trees (e.g. BLEU scores)

        @param trees: A batch of output of evaluation/generation function of the VAE
        @param data: A batch of input for the VAE retrieved from the AstDataset
        @param idx_to_label: Dictionary of ids to labels for the different node types
        """

        scores = {}

        # Get correct formats
        pred_features = [[self.retrieve_features_tree_dfs(tree, idx_to_label, data['nameid_to_placeholderid'][0])] for tree in trees]
        true_features = [f.view(-1).tolist() for f in torch.split(data['features'], data['tree_sizes'])]

        # Compute blue scores
        scores['bleu_4'] = corpus_bleu(pred_features, true_features)
        scores['bleu_3'] = corpus_bleu(pred_features, true_features, weights=(1/3, 1/3, 1/3, 0))
        scores['bleu_2'] = corpus_bleu(pred_features, true_features, weights=(1/2, 1/2, 0, 0))
        scores['bleu_1'] = corpus_bleu(pred_features, true_features, weights=(1, 0, 0, 0))

        return scores


    def retrieve_features_tree_dfs(self, node, idx_to_label, nameid_to_placeholderid):
        """
        Traverse Tree Depth first and put the features in a list. 
        Here the name tokens are transformed to the placeholder IDs.
        This way the features can be compared to the input data to calculate evaluation scores

        @param node: Root Node of the tree
        @param idx_to_label: Dictionary of ids to labels for the different node types
        @param nameid_to_placeholderid: Dictionary of nameids to placeholder ids -> created by AstDataset
        """

        # If the node is a name node, get placeholder ID, otherwise simply append the token
        if not node.res and 'LITERAL' not in idx_to_label['RES'][node.parent.token] and 'TYPE' != idx_to_label['RES'][node.parent.token]:
            features = [nameid_to_placeholderid[node.token]]
        else:
            features = [node.token]
            
        for child in node.children:
            features.extend(self.retrieve_features_tree_dfs(child, idx_to_label, nameid_to_placeholderid))
                
        return features

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
        self.encoder_optimizer.load_state_dict(
            checkpoint['encoder_optimizer_state_dict'])
        self.decoder_optimizer.load_state_dict(
            checkpoint['decoder_optimizer_state_dict'])
        self.losses = checkpoint['losses']
