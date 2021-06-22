import torch
import torch.nn as nn

class AddGate(nn.Module):
    """
        Add gate similar to LSTM add gate: :math: `y = σ(W_mul * inp + b_mul) * tanh(W_add * inp + b_add)`

        Outputs information that can be added to some state
        where the network learns: if and how much of the input should be added
    """

    def __init__(self, dim):
        super().__init__()

        self.W_mul = nn.Linear(dim, dim, bias=True)
        self.W_add = nn.Linear(dim, dim, bias=True)

        self.sigmoid = nn.Sigmoid()


    def forward(self, inp):
        out_mul = self.sigmoid(self.W_mul(inp))
        out_add = torch.tanh(self.W_add(inp))

        return out_mul * out_add


class PredictiveHidden(nn.Module):
    """
        Computes a combined predictive hidden state from two hidden states: :math:`y = tanh(W1 * x1 + W2 * x2)`
    """

    def __init__(self, dim):
        super().__init__()

        # Learnable parameter weights1 -> for calculating: W1 * inp1
        self.W1 = nn.Linear(dim, dim, bias=True)

        # Learnable parameter weights2 -> for calculating: W2 * inp2
        self.W2 = nn.Linear(dim, dim, bias=True)


    def forward(self, inp1, inp2):
        # predictive hidden state: tanh(W1 * inp1 + W2 * inp2)
        h_pred = torch.tanh(self.W1(inp1) + self.W2(inp2))

        return h_pred


class TreeTopologyPred(nn.Module):
    """
        Computes logits for depth, width and res predictions with linear transformations: dim -> 1
    """

    def __init__(self, dim):
        super().__init__()

        # For topology prediction, we predict whether there are children
        self.depth_pred = nn.Linear(dim, 1)

        # For topology prediction, we predict whether there are successor siblings
        self.width_pred = nn.Linear(dim, 1)

        # For predicting whether a token is a reserved keyword of c++ or not
        self.res_pred = nn.Linear(dim, 1)

    def forward(self, inp):
        depth_pred = self.depth_pred(inp)
        width_pred = self.width_pred(inp)
        res_pred = self.res_pred(inp)

        return depth_pred, width_pred, res_pred


class LstmAttention(nn.Module):
    """
        ATTENTION-BASED LSTM FOR PSYCHOLOGICAL STRESS DETECTION FROM SPOKEN
        LANGUAGE USING DISTANT SUPERVISION

        https://arxiv.org/abs/1805.12307
    """

    def __init__(self, dim):
        super().__init__()

        self.attention_weights = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        u = torch.tanh(self.attention_weights(inp))

        a = self.softmax(u)

        v = torch.sum(a * inp, dim=-1)

        return u * inp


class MultiLayerLSTMCell(nn.Module):
    """
        A long short-term memory (LSTM) cell with support for multiple layers.

        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers.
                    E.g., setting num_layers=2 would mean stacking two LSTM cells together
                    to form a stacked LSTM cell, with the second LSTM cell taking in outputs of
                    the first LSTM cell and computing the final results. Default: 1
    """

    def __init__(self, input_size, hidden_size, num_layers = 1, recurrent_dropout=0):
        super().__init__()

        self.num_layers = num_layers
        self.rnns = nn.ModuleList([])
        self.dropout = nn.Dropout(recurrent_dropout)

        # Initialize RNNs with num layers
        for i in range(num_layers):
            if i == 0:
                self.rnns.append(nn.LSTMCell(input_size, hidden_size))
            else:
                self.rnns.append(nn.LSTMCell(hidden_size, hidden_size))


    def forward(self, input, hidden_states):
        new_hidden_states = []

        for i in range(self.num_layers):
            if i == 0:
                h, c = self.rnns[i](input, hidden_states[i])
            else:
                h, c = self.rnns[i](h, hidden_states[i])

            # apply recurrent dropout on the outputs of each LSTM cell hidden except the last layer
            if i < self.num_layers - 1:
                h = self.dropout(h)


            new_hidden_states.append((h, c))

        return new_hidden_states

        

        
