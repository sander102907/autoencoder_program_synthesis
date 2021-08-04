import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeSeqRnnEncoder(nn.Module):
    def __init__(self, device, params, embedding_layers):
        super().__init__()

        self.device = device
        self.params = params
        self.latent_dim = params['LATENT_DIM']

        # Shared embedding layers (shared with encoder)
        self.embedding_layers = embedding_layers

        self.rnn = nn.LSTM(params['EMBEDDING_DIM'], params['HIDDEN_SIZE'] // 4, num_layers=2, bidirectional=True, batch_first=True)

        self.z_mean = nn.Linear(2 * params['HIDDEN_SIZE'], 2 * params['LATENT_DIM'])
        self.z_log_var = nn.Linear(2 * params['HIDDEN_SIZE'], 2 * params['LATENT_DIM'])


    def forward(self, inp):
        node_order = inp['node_order_bottomup']
        vocabs = inp['vocabs']
        batch_size = len(inp['tree_sizes'])

        features = torch.zeros(node_order.shape[0], self.params['EMBEDDING_DIM'], device=self.device)
        
        for k, embedding_layer in self.embedding_layers.items():
            vocab_features = inp['features'][vocabs == k].long()
            emb_vocab_features = embedding_layer(vocab_features).view(-1, self.params['EMBEDDING_DIM'])
            features[vocabs == k] = emb_vocab_features

        
        packed_features = nn.utils.rnn.pack_sequence(features.split(inp['tree_sizes']), enforce_sorted=False)
        _, (hidden, cell) = self.rnn(packed_features)

        output = torch.cat([hidden.view(batch_size, -1), cell.view(batch_size, -1)], dim=-1)

        # Get z_mean and z_log_var from hidden (parent roots only)
        z_mean = self.z_mean(output)
        z_log_var = self.z_log_var(output)
        
        # Parameterization trick
        z = self.reparameterize(z_mean, z_log_var)

        kl_loss = 0.5 * torch.sum(z_log_var.exp() - z_log_var - 1 + z_mean.pow(2))
        
        # Return latent vector, and mean and variance
        return z, kl_loss

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu



        

        

    
