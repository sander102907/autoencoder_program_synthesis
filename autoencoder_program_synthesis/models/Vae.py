import os
import torch
import torch.nn as nn
from autoencoder_program_synthesis.models.TreeSeqRnnEncoder import TreeSeqRnnEncoder
from autoencoder_program_synthesis.model_utils.embeddings import EmbeddingLoader
from autoencoder_program_synthesis.model_utils.metrics_helper import MetricsHelperSeq2Seq, MetricsHelperTree2Tree
from autoencoder_program_synthesis.model_utils.earlystopping import EarlyStopping
from nltk.translate.bleu_score import corpus_bleu
from autoencoder_program_synthesis.utils.TreeNode import Node
from autoencoder_program_synthesis.utils.evaluation import Seq2SeqEvaluator, Tree2TreeEvaluator
from autoencoder_program_synthesis.config.vae_config import ex
from tqdm import tqdm
import numpy as np
import io
import gzip

class Vae(nn.Module):
    """
    A Vae model: wrapper for encoder, decoder models with train, evaluate and generate functions
    Can also be used to easily save and load models
    """

    def __init__(self, device, encoder, decoder, vocabulary, metrics_helper, loss_weights):
        super().__init__()

        self.vocabulary = vocabulary
        self.embedding_layers = nn.ModuleDict({})

        # This only works if we are running a sacred experiment
        try:
        	self.create_embedding_layers()
        	self.encoder = encoder(device, self.embedding_layers)
        	self.decoder = decoder(device, self.embedding_layers, vocabulary, loss_weights)
        except Exception:
            pass

        # Store losses -> so we can easily save them
        self.metrics = {}

        self.metrics_helper = metrics_helper

        self.device = device


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


    @ex.capture
    def create_embedding_layers(self, embedding_dim, indiv_embed_layers, pretrained_emb, pretrained_model):
        # Create shared embedding layers based on vocab sizes we give

        # If we are loading a pretrained model, then we do not need to load pretrained embedding
        if pretrained_model is not None and os.path.isfile(pretrained_model):
            embedding_loader = EmbeddingLoader(embedding_dim)
        else:
            embedding_loader = EmbeddingLoader(embedding_dim, pretrained_emb)

        if indiv_embed_layers:
            for vocab_name in self.vocabulary.token_counts.keys():
                max_tokens = self.vocabulary.get_vocab_size(vocab_name)

                embedding_loader.load_embedding(self.vocabulary.get_cleaned_index2token(vocab_name, max_tokens))
                self.embedding_layers[vocab_name] = embedding_loader.get_embedding_layer()
        else:
            embedding_loader.load_embedding(self.vocabulary.get_cleaned_index2token('ALL'))
            self.embedding_layers['ALL'] = embedding_loader.get_embedding_layer()


    def forward(self, batch):
        z, kl_loss = self.encoder(batch)
        reconstruction_loss, individual_losses, accuracies = self.decoder(z, batch)

        return kl_loss, reconstruction_loss, individual_losses, accuracies


    @ex.capture
    def fit(self, epochs, kl_scheduler, train_loader, val_loader, save_dir=None,
            clip_grad_norm=0, clip_grad_val=0, save_every=1000, check_early_stop_every=500,
            early_stop_patience=3, early_stop_min_delta=0, _run=None):
        """
        Trains the VAE model for the chosen encoder, decoder and its optimizers with the given loss function.
        @param epochs: The number of epochs to train for
        @param train_loader: Torch Dataset that generates batches to train on
        @param val_loader: Torch Dataset that generates batches to validate on
        @param save_dir: The directory to save model checkpoints to, will save every epoch if save_path is given
        """

        print('INFO - Start training..')

        early_stopping = EarlyStopping(early_stop_patience, early_stop_min_delta)
        current_iter_train = 0
        current_iter_val = 0

        self.metrics_helper.init_model_metrics(self.metrics)


        for epoch in range(epochs):
            print('INFO - Starting on epoch: ', epoch)
            # Fit one epoch of training
            current_iter_train, current_iter_val = self._fit_epoch(train_loader, val_loader, kl_scheduler, current_iter_train,
                                                       current_iter_val, clip_grad_norm, clip_grad_val, check_early_stop_every,
                                                       early_stopping, save_every, save_dir, _run)

            # Validate one epoch
            current_iter_val = self._val_epoch(val_loader, current_iter_val, early_stopping, _run)

            if early_stopping.early_stop:
                break


        self.save_model(os.path.join(save_dir, 'model.tar'))

        return self.metrics


    def _fit_epoch(self, train_loader, val_loader, kl_scheduler, current_iter_train, current_iter_val,
                   clip_grad_norm, clip_grad_val, check_early_stop_every, early_stopping,
                   save_every, save_dir, _run):
        self.train()

        for batch_index, batch in enumerate(train_loader):
            current_iteration = current_iter_train + batch_index
            kl_weight = kl_scheduler.get_weight(current_iteration)

            for key in batch.keys():
                if key not in ['ids', 'tree_sizes', 'vocabs', 'declared_names', 'length', 'id']:
                    batch[key] = batch[key].to(self.device)


            kl_loss, reconstruction_loss, individual_losses, accuracies = self(batch)

            loss = (kl_loss * kl_weight + reconstruction_loss)

            loss.backward()

            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)

            if clip_grad_val != 0:
                torch.nn.utils.clip_grad_value_(self.parameters(), clip_grad_val)

            self.optimizer.step()
            self.optimizer.zero_grad()


            if self.metrics_helper == MetricsHelperTree2Tree:
                self.metrics_helper.log_to_sacred(self.training, current_iteration, loss, kl_loss, reconstruction_loss,
                                            individual_losses, accuracies, kl_weight, batch['vocabs'], _run)
                self.metrics_helper.update_model_metrics(self.training, current_iteration, self.metrics, loss,  kl_loss, reconstruction_loss,
                                                individual_losses, accuracies, kl_weight, batch['vocabs'])
            elif current_iteration % 10 == 10 - 1:
                self.metrics_helper.log_to_sacred(self.training, current_iteration, loss, kl_loss, reconstruction_loss,
                                                  kl_weight, 750, _run)
                self.metrics_helper.update_model_metrics(self.training, current_iteration, self.metrics, loss,  kl_loss, reconstruction_loss,
                                                         kl_weight, 750)


            
            # Save model to file every X iterations
            if current_iteration % save_every == save_every - 1 and save_dir is not None:
                print('INFO - Saving model on train iteration: ', current_iteration + 1)
                self.log_model_stats(current_iteration, save_every)

                self.save_model(os.path.join(
                    save_dir, f'iter{current_iteration + 1}.tar'))

            # Check for early stopping every X iterations, run over validation dataset
            if current_iteration % check_early_stop_every == check_early_stop_every - 1:
                current_iter_val = self._val_epoch(val_loader, current_iter_val, early_stopping, _run)
                if early_stopping.early_stop:
                    break

                # after validating, set model back to training mode
                self.train()


        return current_iter_train + batch_index, current_iter_val


    def _val_epoch(self, val_loader, iterations_passed, early_stopping, _run):
        self.eval()

        print('INFO - Going over validation set..')

        total_loss = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                current_iteration = iterations_passed + batch_index

                for key in batch.keys():
                    if key not in ['ids', 'tree_sizes', 'vocabs', 'declared_names', 'length', 'id']:
                        batch[key] = batch[key].to(self.device) 

                z, kl_loss = self.encoder(batch)
                reconstruction_loss, individual_losses, accuracies = self.decoder(z, batch)

                loss = kl_loss + reconstruction_loss
                total_loss += loss.item()

                if self.metrics_helper == MetricsHelperTree2Tree:
                    self.metrics_helper.log_to_sacred(self.training, current_iteration, loss, kl_loss, reconstruction_loss,
                                                individual_losses, accuracies, 1, batch['vocabs'], _run)
                    self.metrics_helper.update_model_metrics(self.training, current_iteration, self.metrics, loss,  kl_loss, reconstruction_loss,
                                                    individual_losses, accuracies, 1, batch['vocabs'])
                else:
                    self.metrics_helper.log_to_sacred(self.training, current_iteration, loss, kl_loss, reconstruction_loss,
                                                    1, 750, _run)
                    self.metrics_helper.update_model_metrics(self.training, current_iteration, self.metrics, loss,  kl_loss, reconstruction_loss,
                                                            1, 750)


        early_stopping(total_loss)
        print('INFO - Total validation loss: ', total_loss)

        return iterations_passed + batch_index


    def test(self, test_loader, temperature, top_k, top_p, save_folder=None, test_programs=None):
        self.eval()
        iterations = 0

        if self.metrics_helper == MetricsHelperTree2Tree:
            evaluator = Tree2TreeEvaluator(self.vocabulary)
        else:
            evaluator = Seq2SeqEvaluator(self.vocabulary)


        with torch.no_grad():
            for batch in tqdm(test_loader):
                iterations += 1
                reconstructions = self.evaluate(batch, temperature, top_k, top_p)

                if self.metrics_helper == MetricsHelperTree2Tree:
                    current_programs = test_programs[test_programs['solutionId'].map(str).isin(batch['ids'])]
                    current_programs['solutionId'] = current_programs['solutionId'].astype(str)
                    current_programs = current_programs.set_index('solutionId')
                    
                    evaluator.add_eval_hypotheses(current_programs.loc[batch['ids']]['solution'], save_folder, batch['ids'])
                    evaluator.add_eval_references(reconstructions, batch['declared_names'])

                    evaluator.reconstructions_to_file(reconstructions, save_folder, batch['ids'])
                else:
                    evaluator.add_eval_hypotheses(batch['input'].tolist(), batch['id'])
                    evaluator.add_eval_references(reconstructions)

                    evaluator.reconstructions_to_file(reconstructions, save_folder, batch['id'])

                # if iterations >= 500:
                    # break

        bleu_scores = evaluator.calc_bleu_score(True)

        # perc_compiles = evaluator.calc_perc_compiles(save_folder, fix_errors=False)
        perc_compiles = 0

        return bleu_scores, perc_compiles


    def evaluate(self, batch, temperature, top_k, top_p):
        """
        Evaluates the VAE model: given data, reconstruct the input and output this
        @param data_loader: Torch Dataset that generates batches to evaluate on
        """

        reconstructions = []

        for key in batch.keys():
            if key not in ['ids', 'tree_sizes', 'vocabs', 'declared_names', 'length', 'id']:
                batch[key] = batch[key].to(self.device)

        z, _ = self.encoder(batch)
        print(z)
        # z = torch.randn([z.shape[0], z.shape[-1]], device=self.device)

        if self.metrics_helper == MetricsHelperTree2Tree:
            reconstructions += self.decoder(z, inp=None, names_token2index=batch['declared_names'], temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            reconstructions += self.decoder(z, inp=None, temperature=temperature, top_k=top_k, top_p=top_p)

        return reconstructions
        
    def encode(self, inp):
        self.eval()
        
        with torch.no_grad():
        	z, _ = self.encoder(inp)

        return z        
        
    def decode(self, z, temperature, top_k, top_p, declared_names):
        self.eval()
        
        with torch.no_grad():
        	output = self.decoder(z, inp=None,temperature=temperature, top_k=top_k, top_p=top_p, generate=True)
        	
        evaluator = Tree2TreeEvaluator(self.vocabulary)
        program = evaluator.reconstructions_to_code(output, True, declared_names)
        
        return program


    def generate(self, z, save_folder, temperature, top_k, top_p):
        """
        Generate using the VAE model: given latent vector(s) z, generate output
        @param z: latent vector(s) -> (batch_size, latent_size)
        """

        self.eval()

        if self.metrics_helper == MetricsHelperTree2Tree:
            for batch in torch.split(z, 32):
                with torch.no_grad():
                    output = self.decoder(batch, inp=None, temperature=temperature, top_k=top_k, top_p=top_p, generate=True)

                evaluator = Tree2TreeEvaluator(self.vocabulary)
                evaluator.generations_to_file(output, save_folder)
        else:
            for batch in torch.split(z, 5):
                with torch.no_grad():
                    output = self.decoder(batch, inp=None, temperature=temperature, top_k=top_k, top_p=top_p)

                evaluator = Seq2SeqEvaluator(self.vocabulary)
                evaluator.generations_to_file(output, save_folder)
            

        return output


    def interpolate(self, test_loader, n, save_folder, temperature, top_k, top_p):
        self.eval()


        for batch in test_loader:
            for key in batch.keys():
                if key not in ['ids', 'tree_sizes', 'vocabs', 'declared_names', 'length', 'id']:
                    batch[key] = batch[key].to(self.device)


        with torch.no_grad():
            print(batch)
            z, _ = self.encoder(batch)

            z = torch.stack([z[0] + (z[1] - z[0]) * t for t in np.linspace(0, 1, n)])
            interpolate_list = self.decoder(z, inp=None, temperature=temperature, top_k=top_k, top_p=top_p, generate=True)



        if self.metrics_helper == MetricsHelperTree2Tree:
            evaluator = Tree2TreeEvaluator(self.vocabulary)
            evaluator.generations_to_file(interpolate_list, save_folder, plugin_names=True)
        else:
            evaluator = Seq2SeqEvaluator(self.vocabulary)
            evaluator.generations_to_file(interpolate_list, save_folder)





    def calculate_eval_scores(self, trees, data):
        """
        Calculate evaluation scores for generated trees (e.g. BLEU scores)

        @param trees: A batch of output of evaluation/generation function of the VAE
        @param data: A batch of input for the VAE retrieved from the AstDataset
        @param idx_to_label: Dictionary of ids to labels for the different node types
        """

        scores = {}

        # Get correct formats
        pred_features = [[self.retrieve_features_tree_dfs(tree, data['declared_names'][index].names)] for index, tree in enumerate(trees)]
        true_features = [f.view(-1).tolist() for f in torch.split(data['features'], data['tree_sizes'])]

        # Compute blue scores
        scores['bleu_4'] = corpus_bleu(pred_features, true_features)
        scores['bleu_3'] = corpus_bleu(pred_features, true_features, weights=(1/3, 1/3, 1/3, 0))
        scores['bleu_2'] = corpus_bleu(pred_features, true_features, weights=(1/2, 1/2, 0, 0))
        scores['bleu_1'] = corpus_bleu(pred_features, true_features, weights=(1, 0, 0, 0))

        return scores


    def retrieve_features_tree_dfs(self, node, oov_name_token2index):
        """
        Traverse Tree Depth first and put the features in a list. 
        Here the name tokens are transformed to the placeholder IDs.
        This way the features can be compared to the input data to calculate evaluation scores

        @param node: Root Node of the tree
        @param idx_to_label: Dictionary of ids to labels for the different node types
        @param nameid_to_placeholderid: Dictionary of nameids to placeholder ids -> created by AstDataset
        """

        # If the node is a name node, get placeholder ID, otherwise simply append the token
        try:
            if not node.res and 'LITERAL' not in self.vocabulary.index2token['RES'][node.parent.token] and 'TYPE' != self.vocabulary.index2token['RES'][node.parent.token] and node.token in oov_name_token2index:
                features = [oov_name_token2index[node.token]]
            else:
                features = [node.token]
        except KeyError:
            features = [node.token]
            
        for child in node.children:
            features.extend(self.retrieve_features_tree_dfs(child, oov_name_token2index))
                
        return features


    def log_model_stats(self, current_iteration, save_every):
        print('INFO - Model statistics:')
        if self.metrics_helper == MetricsHelperTree2Tree:
            print('\t Avg Total loss / node: ', sum(list(self.metrics['Total loss / node train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc parent: ', sum(list(self.metrics['Parent accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc sibling: ', sum(list(self.metrics['Sibling accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc is reserved: ', sum(list(self.metrics['Is reserved accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc reserved label: ', sum(list(self.metrics['Reserved label accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc is type label: ', sum(list(self.metrics['Type label accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc literal: ', sum(list(self.metrics['Literal label accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc name builtin: ', sum(list(self.metrics['Name builtin label accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
            print('\t Avg acc name: ', sum(list(self.metrics['Name label accuracy train'].values())[current_iteration - save_every: current_iteration])/save_every)
        else:
            print('\t Avg Total loss / word: ', sum(list(self.metrics['Total loss / word train'].values())[current_iteration - save_every: current_iteration])/save_every)


    
    def build_tree(self, adj_list, features, vocabs, index=0, parent_node=None):
        node = Node(features[index].item(), is_reserved=vocabs[index] == 'RES', parent=parent_node)
        children = adj_list[adj_list[:, 0] == index][:, 1]

        for child in children:
            self.build_tree(adj_list, features, vocabs, child, node)

        return node

    def save_model(self, path):
        """
        Save the encoder, decoder and corresponding optimizer state dicts as well as losses to .tar
        Such that the model can be used to continue training or for inference
        @param path: Save the model to the given path
        """

        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)

        torch.save({
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)



    def load_model(self, path):
        """
        Load a model save to the encoder, decoder and corresponding optimizer state dicts and load losses
        Can be used to continue training a model or load a model for inference
        @param path: Load a model from the given path
        """

        if '.gz' in path:
            with gzip.open(path, 'rb') as f:
                x = io.BytesIO(f.read())
                checkpoint = torch.load(x, map_location=self.device)
        else:
            checkpoint = torch.load(path, map_location=self.device)

        try:
            print(f'INFO - Loading weights from model: {path}')
            self.load_state_dict(checkpoint['state_dict'])
        except RuntimeError as e:
            print(f'INFO - {e}')
            print('INFO - skipping missing key(s), loading the rest of the model weights')
            self.load_state_dict(checkpoint['state_dict'], strict=False)

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except AttributeError as e:
            print(f'INFO - skipped loading optimizer state dict: \n {e}')
        except ValueError as e:
            print(f'INFO - skipped loading optimizer state dict: \n {e}')

        try:
            self.metrics = checkpoint['metrics']
        except KeyError:
            print('INFO - skip loading metrics, not available in pretrained model')
