import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from config.vae_config import ex


"""
https://github.com/timbmg/Sentence-VAE
https://arxiv.org/abs/1511.06349

"""

class SeqRnnEncoder(nn.Module):
    @ex.capture
    def __init__(self,
                device,
                embedding_layers,
                embedding_dim,
                rnn_hidden_size,
                latent_dim,
                vae,
                num_rnn_layers_enc,
                recurrent_dropout,
                bidirectional=False):

        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.vae = vae
        self.bidirectional = bidirectional
        self.num_rnn_layers = num_rnn_layers_enc
        self.rnn_hidden_size = rnn_hidden_size

        self.embedding_layer = embedding_layers['ALL']

        self.rnn = nn.RNN(embedding_dim,
                          rnn_hidden_size,
                          num_layers=num_rnn_layers_enc, 
                          bidirectional=bidirectional, 
                          batch_first=True, 
                          dropout=recurrent_dropout)

        self.hidden_factor = (2 if bidirectional else 1) * num_rnn_layers_enc

        self.z_mean = nn.Linear(rnn_hidden_size * self.hidden_factor, latent_dim)
        self.z_log_var = nn.Linear(rnn_hidden_size * self.hidden_factor, latent_dim)


    def forward(self, inp):
        inp_seq = inp['input']
        length = inp['length']

        batch_size = inp_seq.shape[0]
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        inp_seq = inp_seq[sorted_idx]

        inp_emb = self.embedding_layer(inp_seq)
        packed_inp = rnn_utils.pack_padded_sequence(inp_emb, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.rnn(packed_inp)

        if self.bidirectional or self.num_rnn_layers > 1:
            # Flatten the hidden state
            hidden = hidden.view(batch_size, self.rnn_hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()


        # Get z_mean and z_log_var from hidden (parent roots only)
        z_mean = self.z_mean(hidden)
        z_log_var = self.z_log_var(hidden)

        
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


