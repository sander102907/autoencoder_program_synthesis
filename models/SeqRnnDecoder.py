import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils.Sampling import Sampling

class SeqRnnDecoder(nn.Module):
    def __init__(self,
                device,
                embedding_layer,
                embedding_dim,
                rnn_hidden_size,
                latent_dim,
                num_rnn_layers,
                recurrent_dropout,
                vocab_size,
                bidirectional=False,
                max_sequence_length=1500):

        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.max_sequence_length = max_sequence_length

        self.embedding_layer = embedding_layer

        self.rnn = nn.RNN(embedding_dim,
                          rnn_hidden_size,
                          num_layers=num_rnn_layers, 
                          bidirectional=bidirectional, 
                          batch_first=True, 
                          dropout=recurrent_dropout)

        self.hidden_factor = (2 if bidirectional else 1) * num_rnn_layers

        self.latent2hidden = nn.Linear(latent_dim, rnn_hidden_size * self.hidden_factor)
        self.prediction_layer = nn.Linear(rnn_hidden_size * (2 if bidirectional else 1), vocab_size)

        self.loss = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, z, inp):
        if inp is not None:
            return self.forward_train(z, inp)
        else:
            return self.forward_inference(z)
        

    def forward_train(self, z, inp):
        batch_size = inp.shape[0]
        sorted_lengths, sorted_idx = None

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_rnn_layers > 1:
            # Unflatten the hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.rnn_hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        inp_emb = self.embedding_layer(inp)
        packed_inp = rnn_utils.pack_padded_sequence(inp_emb, sorted_lengths.data.tolist(), batch_first=True)

        outputs, _ = self.rnn(packed_inp, hidden)

        # Process RNN outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()


        # Get predictions
        logits = self.prediction_layer(padded_outputs.view(-1, padded_outputs.size(2)))
        loss = self.loss(logits)

        return loss


    def forward_inference(self, z):
        batch_size = z.shape[0]

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_rnn_layers > 1:
            # Unflatten the hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.rnn_hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        
         # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                sequence = torch.Tensor(size=batch_size, fill_value=self.sos_idx, dtype=torch.long, device=self.device)

            sequence = sequence.unsqueeze(1)

            seq_emb = self.embedding(sequence)

            output, hidden = self.decoder_rnn(seq_emb, hidden)

            logits = self.prediction_layer(output)

            pred_seq = Sampling.sample(logits, temperature=0.7, top_k=40, top_p=0.9).view(-1)


            # update gloabl running sequence
            sequence_mask[sequence_running] = (pred_seq != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (pred_seq != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                pred_seq = pred_seq[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations


