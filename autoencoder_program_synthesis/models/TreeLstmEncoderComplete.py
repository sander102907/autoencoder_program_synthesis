import torch
import torch.nn as nn
from models.tree_lstm import TreeLSTM
from model_utils.modules import LstmAttention
from config.vae_config import ex


class TreeLstmEncoderComplete(nn.Module):
    @ex.capture
    def __init__(self, 
                device, 
                embedding_layers, 
                embedding_dim, 
                rnn_hidden_size, 
                latent_dim, 
                use_cell_output_lstm, 
                vae,
                num_rnn_layers_enc,
                recurrent_dropout,
                indiv_embed_layers):

        super().__init__() 
        
        self.device = device
        self.embedding_dim = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.vae = vae
        self.num_rnn_layers_enc = num_rnn_layers_enc
        self.indiv_embed_layers = indiv_embed_layers
        self.use_cell_output_lstm = use_cell_output_lstm
        

        self.embedding_layers = embedding_layers
        self.tree_lstms = nn.ModuleList([])
        self.leaf_lstms = nn.ModuleDict({})
        self.attention = LstmAttention(rnn_hidden_size)

        self.recurrent_dropout = nn.Dropout(recurrent_dropout)

        for i in range(num_rnn_layers_enc):
            if i == 0:
                self.tree_lstms.append(TreeLSTM(embedding_dim, rnn_hidden_size))
            else:
                self.tree_lstms.append(TreeLSTM(rnn_hidden_size, rnn_hidden_size))


        if use_cell_output_lstm:
            rnn_hidden_size = rnn_hidden_size * 2
            latent_dim = latent_dim * 2

        self.z_mean = nn.Linear(rnn_hidden_size, latent_dim)
        self.z_log_var = nn.Linear(rnn_hidden_size, latent_dim)

        
    def forward(self, inp):
        batch_size = len(inp['tree_sizes'])
        node_order = inp['node_order_bottomup']
        adj_list = inp['adjacency_list']
        edge_order = inp['edge_order_bottomup']
        vocabs = inp['vocabs']
        total_nodes = node_order.shape[0]

        features = torch.zeros(total_nodes, self.embedding_dim, device=self.device)
        h = []
        c = []

        for _ in range(self.num_rnn_layers_enc):
            h.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))
            c.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))

        
        if self.indiv_embed_layers:
            for k, embedding_layer in self.embedding_layers.items():
                features_vocab = embedding_layer(inp['features'][vocabs == k].long()).view(-1, self.embedding_dim)
                features[vocabs == k] = features_vocab

        else:
            features_vocab = self.embedding_layers['ALL'](inp['features_combined'].long()).view(-1, self.embedding_dim)
            features = features_vocab

        for i in range(self.num_rnn_layers_enc):
            hidden, cell = self.tree_lstms[i](features,
                                        node_order,
                                        adj_list,
                                        edge_order,
                                        h[i],
                                        c[i])

            if i < self.num_rnn_layers_enc - 1:
                hidden = self.recurrent_dropout(hidden)

            features = hidden
        
        
        # hidden roots: (batch_size, hidden_size (*2 if using cell state of LSTM output))
        hidden_roots = torch.zeros(batch_size,
                                   self.rnn_hidden_size *2 if self.use_cell_output_lstm else self.rnn_hidden_size,
                                   device=self.device)

        hidden = self.attention(hidden)

        
        # Offset to check in hidden state, start at zero, increase by tree size each time
        # Example: hidden  [1, 3, 5, 1, 5, 2] and tree_sizes = [4, 2] we want hidden[0] and hidden[4] -> 1, 5
        # offset = 0
        # for i in range(len(inp['tree_sizes'])):
        #     if self.use_cell_output_lstm:
        #         hidden_roots[i] = torch.cat([hidden[offset], cell[offset]])
        #     else:
        #         hidden_roots[i] = hidden[offset]
        #     offset += inp['tree_sizes'][i]

        # tree_states = torch.cat([hidden, cell], dim=-1)

        for idx, tree_state in enumerate(torch.split(hidden, inp['tree_sizes'])):
            hidden_roots[idx] = torch.max(tree_state, dim=0)[0]
            # hidden_roots[idx] = tree_state[-1]

        # Get z_mean and z_log_var from hidden (parent roots only)
        z_mean = self.z_mean(hidden_roots)
        z_log_var = self.z_log_var(hidden_roots)

        
        # Parameterization trick
        z = self.reparameterize(z_mean, z_log_var)


        if self.vae:
            kl_loss = (0.5 * torch.sum(z_log_var.exp() - z_log_var - 1 + z_mean.pow(2)))
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
