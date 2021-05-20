import torch
import torch.nn as nn
from models.tree_lstm import TreeLSTM
from model_utils.modules import LstmAttention

class TreeLstmEncoderComplete(nn.Module):
    def __init__(self, device, params, embedding_layers):
        super().__init__() 
        
        self.device = device
        self.params = params
        self.hidden_size = params['HIDDEN_SIZE']
        self.embedding_layers = embedding_layers
        self.tree_lstms = nn.ModuleList([])
        self.leaf_lstms = nn.ModuleDict({})
        self.vae = self.params['VAE']
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.attention = LstmAttention(self.hidden_size)

        for i in range(self.params['NUM_LSTM_LAYERS']):
            if i == 0:
                self.tree_lstms.append(TreeLSTM(params['EMBEDDING_DIM'], params['HIDDEN_SIZE']))
            else:
                self.tree_lstms.append(TreeLSTM(params['HIDDEN_SIZE'], params['HIDDEN_SIZE']))

        if params['INDIV_LAYERS_VOCABS']:
            for k, embedding_layer in embedding_layers.items():
                if not 'RES' in k:
                    self.leaf_lstms[k] = nn.LSTMCell(params['LEAF_EMBEDDING_DIM'], params['HIDDEN_SIZE'])


        if params['USE_CELL_LSTM_OUTPUT']:
            hidden_size = params['HIDDEN_SIZE'] * 2
            latent_dim = params['LATENT_DIM'] * 2
        else:
            hidden_size = params['HIDDEN_SIZE']
            latent_dim = params['LATENT_DIM']

        self.z_mean = nn.Linear(hidden_size, latent_dim)
        self.z_log_var = nn.Linear(hidden_size, latent_dim)
        
    def forward(self, inp):
        batch_size = len(inp['tree_sizes'])
        node_order = inp['node_order_bottomup']
        adj_list = inp['adjacency_list']
        edge_order = inp['edge_order_bottomup']
        vocabs = inp['vocabs']

        features = torch.zeros(node_order.shape[0], self.params['EMBEDDING_DIM'], device=self.device)
        h = []
        c = []

        for _ in range(self.params['NUM_LSTM_LAYERS']):
            h.append(torch.zeros(node_order.shape[0], self.hidden_size, device=self.device))
            c.append(torch.zeros(node_order.shape[0], self.hidden_size, device=self.device))
        
        for k, embedding_layer in self.embedding_layers.items():
            if self.params['INDIV_LAYERS_VOCABS']:
                if 'RES' in k:
                    features_vocab = embedding_layer(inp['features'][vocabs == k].long()).view(-1, self.params['EMBEDDING_DIM'])
                    features[vocabs == k] = features_vocab

                else:
                    features_vocab = embedding_layer(inp['features'][vocabs == k].long()).view(-1, self.params['LEAF_EMBEDDING_DIM'])
                    h_vocab, c_vocab = self.leaf_lstms[k](features_vocab)
                    h[vocabs == k] = h_vocab
                    c[vocabs == k] = c_vocab
            else:
                features_vocab = embedding_layer(inp['features'][vocabs == k].long()).view(-1, self.params['EMBEDDING_DIM'])
                features[vocabs == k] = features_vocab

        # Tree LSTM can start from bottom nodes (iteration 0) if 1 LSTM is used, if individual leaf LSTMS are used, start at iteration 1
        start_iteration = 1 if self.params['INDIV_LAYERS_VOCABS'] else 0

        for i in range(self.params['NUM_LSTM_LAYERS']):
            hidden, cell = self.tree_lstms[i](features,
                                        node_order,
                                        adj_list,
                                        edge_order,
                                        h[i],
                                        c[i],
                                        start_iteration)

            features = hidden
        
        
        # hidden roots: (batch_size, hidden_size (*2 if using cell state of LSTM output))
        hidden_roots = torch.zeros(batch_size,
                                   self.hidden_size *2 if self.params['USE_CELL_LSTM_OUTPUT'] else self.hidden_size,
                                   device=self.device)

        # hidden = self.attention(hidden)

        
        # Offset to check in hidden state, start at zero, increase by tree size each time
        # Example: hidden  [1, 3, 5, 1, 5, 2] and tree_sizes = [4, 2] we want hidden[0] and hidden[4] -> 1, 5
        # offset = 0
        # for i in range(len(inp['tree_sizes'])):
        #     if self.params['USE_CELL_LSTM_OUTPUT']:
        #         hidden_roots[i] = torch.cat([hidden[offset], cell[offset]])
        #     else:
        #         hidden_roots[i] = hidden[offset]
        #     offset += inp['tree_sizes'][i]

        # tree_states = torch.cat([hidden, cell], dim=-1)

        for idx, tree_state in enumerate(torch.split(hidden, inp['tree_sizes'])):
            hidden_roots[i] = torch.max(tree_state, dim=0)[0]

        # Get z_mean and z_log_var from hidden (parent roots only)
        z_mean = self.z_mean(hidden_roots)
        z_log_var = self.z_log_var(hidden_roots)

        
        # Parameterization trick
        z = self.reparameterize(z_mean, z_log_var)

        if self.vae:
            kl_loss = 0.5 * torch.sum(z_log_var.exp() - z_log_var - 1 + z_mean.pow(2))
        else:
            kl_loss = torch.tensor([0], device=self.device)

        
        # Return latent vector, and mean and variance
        return z, kl_loss
        
        
    def reparameterize(self, mu, log_var):
        if self.training and self.vae:
            std = torch.exp(0.5 * log_var)
            eps = std.data.new(std.size()).normal_()

            return eps.mul(std).add_(mu)
        else:
            return mu