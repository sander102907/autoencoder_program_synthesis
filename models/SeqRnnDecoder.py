import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from model_utils.adaptive_softmax_pytorch import AdaptiveLogSoftmaxWithLoss
from utils.Sampling import Sampling
from config.vae_config import ex
import time

class SeqRnnDecoder(nn.Module):
    @ex.capture
    def __init__(self,
                device,
                embedding_layers,
                vocabulary,
                loss_weights,
                embedding_dim,
                rnn_hidden_size,
                latent_dim,
                num_rnn_layers_dec,
                recurrent_dropout,
                max_program_size,
                word_dropout=0.25,
                rnn_type='lstm',
                ):

        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.num_rnn_layers = num_rnn_layers_dec
        self.rnn_hidden_size = rnn_hidden_size
        self.max_program_size = max_program_size
        self.vocab_size = vocabulary.get_vocab_size('ALL')
        self.rnn_type = rnn_type
        self.word_dropout = word_dropout # replace percentage of tokens with unk token to make model depend more on latent variable
        print(self.vocab_size)

        self.embedding_layer = embedding_layers['ALL']

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM

        self.rnn = rnn(embedding_dim,
                          rnn_hidden_size,
                          num_layers=num_rnn_layers_dec, 
                          batch_first=True, 
                          dropout=recurrent_dropout)

        self.hidden_factor = num_rnn_layers_dec

        self.latent2hidden = nn.Linear(latent_dim, rnn_hidden_size * self.hidden_factor)


        # Adaptive softmax loss does not need prediction layer, much faster approach
        # for calculating softmax with highly imbalanced vocabs
        # cutoffs: 100: 90.2%, 101-1000: 7.9%, 1001-10000: 1.3%, rest: 0.4%
        self.loss = AdaptiveLogSoftmaxWithLoss(
                                            self.rnn_hidden_size,
                                            self.vocab_size,
                                            cutoffs=[100, 1000, 10000],
                                            div_value=2.0)


        self.sos_idx = vocabulary.token2index['ALL']['<sos>']
        self.eos_idx = vocabulary.token2index['ALL']['<eos>']
        self.pad_idx = vocabulary.token2index['ALL']['<pad>']
        self.unk_idx = vocabulary.token2index['ALL']['<unk>']


    def forward(self, z, inp=None, temperature=None, top_k=None, top_p=None, generate=False):
        if inp is not None:
            return self.forward_train(z, inp)
        else:
            return self.forward_inference(z, temperature, top_k, top_p)
        

    def forward_train(self, z, inp):
        inp_seq = inp['input']
        target_seq = inp['target']
        length = inp['length']

        batch_size = inp_seq.shape[0]
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        inp_seq = inp_seq[sorted_idx]

        hidden = self.latent2hidden(z)

        if self.num_rnn_layers > 1:
            # Unflatten the hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.rnn_hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        if self.rnn_type == 'lstm':
            cell = torch.zeros((self.hidden_factor, batch_size, self.rnn_hidden_size), device=self.device)
            hidden = (hidden, cell)


        if self.word_dropout > 0 and self.training:
            # randomly replace decoder input with <unk>
            prob = torch.rand(inp_seq.size(), device=self.device)
            prob[(inp_seq.data - self.sos_idx) * (inp_seq.data - self.pad_idx) == 0] = 1
            inp_seq[prob < self.word_dropout] = self.unk_idx


        inp_emb = self.embedding_layer(inp_seq)

        packed_inp = rnn_utils.pack_padded_sequence(inp_emb, sorted_lengths.data.tolist(), batch_first=True)

        outputs, _ = self.rnn(packed_inp, hidden)

        # Process RNN outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True, padding_value=self.pad_idx)[0]

        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        _,s,_ = padded_outputs.size()

        # cut-off unnessary padding from target and flatten
        target_seq = target_seq[:, :s].contiguous().view(-1)

        # "flatten" output as well to (B * max_seq_len, hidden_size)
        padded_outputs = padded_outputs.view(-1, self.rnn_hidden_size)

        outputs, loss = self.loss(padded_outputs, target_seq)

        # We do not want to punish the model for predicting padding incorrectly
        # Therefore we set the loss for the paddings to 0
        loss[target_seq == self.pad_idx] = 0

        # We sum to get equal weights with KL loss
        loss = torch.sum(loss)

        return loss, {}, {}


    def forward_inference(self, z, temperature, top_k, top_p):
        batch_size = z.shape[0]

        hidden = self.latent2hidden(z)

        if self.num_rnn_layers > 1:
            # Unflatten the hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.rnn_hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        if self.rnn_type == 'lstm':
            cell = torch.zeros((self.hidden_factor, batch_size, self.rnn_hidden_size), device=self.device)
            hidden = (hidden, cell)

        
         # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, device=self.device, dtype=torch.long) # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, device=self.device, dtype=torch.long)
        sequence_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, device=self.device, dtype=torch.long)

        generations = torch.full(fill_value=self.pad_idx, size=(batch_size, self.max_program_size), device=self.device, dtype=torch.long)

        t = 0
        while t < self.max_program_size and len(running_seqs) > 0:

            if t == 0:
                sequence = torch.full(size=(batch_size,), fill_value=self.sos_idx, dtype=torch.long, device=self.device)

            sequence = sequence.unsqueeze(1)

            seq_emb = self.embedding_layer(sequence)

            output, hidden = self.rnn(seq_emb, hidden)

            logits = self.loss.log_prob(output.squeeze(1))

            sequence = Sampling.sample(logits, temperature, top_k, top_p).view(-1)

            # Save the next sequence that will be input to the RNN
            generations = self._save_sample(generations, sequence, sequence_running, t)


            # update global running sequence
            sequence_mask[sequence_running] = (sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                sequence = sequence[running_seqs]

                if self.rnn_type == 'lstm':
                    hidden = (hidden[0][:, running_seqs], hidden[1][:, running_seqs])
                else:
                    hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), device=self.device, dtype=torch.long)

            t += 1

        return generations


    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to


