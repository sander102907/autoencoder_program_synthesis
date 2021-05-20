import torch
import numpy as np

class TreeState:
    def __init__(self, target, params, device):
        # Get input data
        self.node_order = target['node_order_topdown']
        self.edge_order = target['edge_order_topdown']
        self.features = target['features']
        self.adj_list = target['adjacency_list']
        self.vocabs = target['vocabs']
        self.total_nodes = self.node_order.shape[0]

        # Parameters
        self.params = params

        # Device
        self.device = device

        # Hidden and cell states of RNNs for parent and sibling
        self.h_p = []
        self.c_p = []

        self.h_s = []
        self.c_s = []

        # Initialize hidden and cell states
        self.init_states()

        # Current tree state properties
        self.h_parent = None
        self.c_parent = None
        self.h_prev_sibling = None
        self.c_prev_sibling = None
        self.is_parent = None
        self.has_sibling = None
        self.current_nodes_indices = None
        self.vocabs_mask = None
        self.is_res = None


    def init_states(self):
        for _ in range(self.params['NUM_LSTM_LAYERS']):
            # h and c states for every node in the batch for parent lstm
            self.h_p.append(torch.zeros(self.total_nodes, self.params['LATENT_DIM'], device=self.device))
            self.c_p.append(torch.zeros(self.total_nodes, self.params['LATENT_DIM'], device=self.device))

            # h and c states for every node in the batch for sibling lstm
            self.h_s.append(torch.zeros(self.total_nodes, self.params['LATENT_DIM'], device=self.device))
            self.c_s.append(torch.zeros(self.total_nodes, self.params['LATENT_DIM'], device=self.device))


    def update_hidden_values(self, iteration, sibling_index, first_sibling_indices, parent_indices):
        # Get adjacency list of previous iteration
        adj_list_prev = self.adj_list[self.edge_order == iteration - 1, :]

        # Get the indices of the current nodes
        current_indices = adj_list_prev[:, 1]

        # At sibling index 0, there should not be any previous siblings
        if sibling_index == 0:
            num_first_siblings = len(first_sibling_indices)
            # Initialize to hidden, cell of siblings to zero
            self.h_prev_sibling = []
            self.c_prev_sibling = []

            init_state = torch.zeros(num_first_siblings, self.params['LATENT_DIM'], device=self.device)
            for i in range(self.params['NUM_LSTM_LAYERS']):
                self.h_prev_sibling.append(init_state)
                self.c_prev_sibling.append(init_state)

            self.current_nodes_indices = current_indices[first_sibling_indices]
            parent_indices_siblings = parent_indices[first_sibling_indices]

        else:
            indices = []

            for i, first_sibling_index in enumerate(first_sibling_indices):
                if (i + 1 < len(first_sibling_indices)
                    and sibling_index + first_sibling_index < first_sibling_indices[i + 1])\
                    or (i + 1 == len(first_sibling_indices) and sibling_index + first_sibling_index < len(current_indices)):
                    
                    indices.append(first_sibling_index + sibling_index - 1)

            prev_siblings_indices = current_indices[indices]

            self.h_prev_sibling = []
            self.c_prev_sibling = []

            for i in range(self.params['NUM_LSTM_LAYERS']):
                self.h_prev_sibling.append(self.h_s[i][prev_siblings_indices, :])
                self.c_prev_sibling.append(self.c_s[i][prev_siblings_indices, :])

            parent_indices_siblings = parent_indices[indices]
            self.current_nodes_indices = current_indices[[
                ind + 1 for ind in indices]]

        self.h_parent = []
        self.c_parent = []

        for i in range(self.params['NUM_LSTM_LAYERS']):
            self.h_parent.append(self.h_p[i][parent_indices_siblings, :])
            self.c_parent.append(self.c_p[i][parent_indices_siblings, :])

        adj_list_curr = self.adj_list[self.edge_order == iteration, :]
        sib = list(first_sibling_indices) + [len(parent_indices)]

        self.vocabs_mask = np.atleast_1d(self.vocabs[self.current_nodes_indices.cpu()])

        self.is_parent = torch.tensor([[1.] if index in adj_list_curr[:, 0] else [
                                 0.] for index in self.current_nodes_indices], device=self.device)

        self.has_sibling = torch.tensor([[1.] if j-i - 1 > sibling_index else [0.]
                                    for i,  j in zip(sib[:-1], sib[1:]) if j-i > sibling_index], device=self.device)

        self.is_res = torch.tensor(self.vocabs_mask == 'RES', dtype=torch.float, device=self.device).view(-1, 1)
