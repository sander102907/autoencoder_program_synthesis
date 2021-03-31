import torch
import torch.nn as nn
from treelstm import TreeLSTM

class TreeLstmEncoder(nn.Module):
    def __init__(self, device, params, embedding):
        super().__init__() 
        
        self.device = device
                
        self.hidden_size = params['HIDDEN_SIZE']
        self.embedding_dim = params['EMBEDDING_DIM']
        
        self.embedding = embedding
        self.tree_lstm = TreeLSTM(params['EMBEDDING_DIM'], params['HIDDEN_SIZE'])
        self.z_mean = nn.Linear(params['HIDDEN_SIZE'], params['LATENT_DIM'])
        self.z_log_var = nn.Linear(params['HIDDEN_SIZE'], params['LATENT_DIM'])
        
    def forward(self, inp):
        batch_size = len(inp['tree_sizes'])
        
        features = self.embedding(inp['features'].long()).view(-1, self.embedding_dim)
        hidden, cell = self.tree_lstm(features,
                                      inp['node_order_bottomup'],
                                      inp['adjacency_list'],
                                      inp['edge_order_bottomup'])
        
        
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