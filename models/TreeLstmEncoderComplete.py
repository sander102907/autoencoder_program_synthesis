import torch
import torch.nn as nn
# from treelstm import TreeLSTM
from models.tree_lstm import TreeLSTM

class TreeLstmEncoderComplete(nn.Module):
    def __init__(self, device, params, embedding_layers):
        super().__init__() 
        
        self.device = device
        self.params = params
        self.hidden_size = params['HIDDEN_SIZE']
        self.embedding_layers = embedding_layers
        self.tree_lstm = TreeLSTM(params['EMBEDDING_DIM'], params['HIDDEN_SIZE'])
        self.leaf_lstms = nn.ModuleDict({})

        for k, embedding_layer in embedding_layers.items():
            if not 'RES' in k:
                self.leaf_lstms[k] = nn.LSTMCell(params['LEAF_EMBEDDING_DIM'], params['HIDDEN_SIZE'])

        self.z_mean = nn.Linear(params['HIDDEN_SIZE'], params['LATENT_DIM'])
        self.z_log_var = nn.Linear(params['HIDDEN_SIZE'], params['LATENT_DIM'])
        
    def forward(self, inp):
        batch_size = len(inp['tree_sizes'])
        node_order = inp['node_order_bottomup']
        adj_list = inp['adjacency_list']
        edge_order = inp['edge_order_bottomup']
        vocabs = inp['vocabs']

        features = torch.zeros(node_order.shape[0], self.params['EMBEDDING_DIM'], device=self.device)
        h = torch.zeros(node_order.shape[0], self.hidden_size, device=self.device)
        c = torch.zeros(node_order.shape[0], self.hidden_size, device=self.device)
        
        for k, embedding_layer in self.embedding_layers.items():
            if 'RES' in k:
                features_vocab = embedding_layer(inp['features'][vocabs == k].long()).view(-1, self.params['EMBEDDING_DIM'])
                features[vocabs == k] = features_vocab

            else:
                features_vocab = embedding_layer(inp['features'][vocabs == k].long()).view(-1, self.params['LEAF_EMBEDDING_DIM'])
                h_vocab, c_vocab = self.leaf_lstms[k](features_vocab)
                h[vocabs == k] = h_vocab
                c[vocabs == k] = c_vocab
        

        hidden, cell = self.tree_lstm(features,
                                      node_order,
                                      adj_list,
                                      edge_order,
                                      h,
                                      c)
        
        
        # Take hidden states of roots of trees only -> tree lstm produces hidden states for all nodes in all trees as list
        # hidden roots: (batch_size, hidden_size)
        hidden_roots = torch.zeros(batch_size, self.hidden_size, device=self.device)
        
        # Offset to check in hidden state, start at zero, increase by tree size each time
        # Example: hidden  [1, 3, 5, 1, 5, 2] and tree_sizes = [4, 2] we want hidden[0] and hidden[4] -> 1, 5
        offset = 0
        for i in range(len(inp['tree_sizes'])):
            hidden_roots[i] = hidden[offset]
            offset += inp['tree_sizes'][i]

        # Get z_mean and z_log_var from hidden (parent roots only)
        z_mean = self.z_mean(hidden_roots)
        z_log_var = self.z_log_var(hidden_roots)
        
        # Parameterization trick
        z = self.reparameterize(z_mean, z_log_var)
        
        # Return latent vector, and mean and variance
        return z, z_mean, z_log_var
        
        
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu