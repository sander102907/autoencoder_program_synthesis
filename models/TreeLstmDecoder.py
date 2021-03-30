import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.TreePredictionNode import Node

class TreeLstmDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, device):
        super().__init__()
        
        self.device = device
        
        self.vocab_size = vocab_size
        self.latent_size = latent_size
        
        self.lstm_parent = nn.LSTMCell(vocab_size, latent_size)
        self.U_parent = nn.Linear(latent_size, latent_size, bias=False)
        self.depth_pred = nn.Linear(latent_size, 1)
        
        self.lstm_sibling = nn.LSTMCell(vocab_size, latent_size)
        self.U_sibling = nn.Linear(latent_size, latent_size, bias=False)
        self.width_pred = nn.Linear(latent_size, 1)
        
        self.label_prediction = nn.Linear(latent_size, vocab_size)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.offset_parent = nn.Linear(1, 1, bias=False)
        self.offset_sibling = nn.Linear(1, 1, bias=False)
        
        
    def forward(self, z, target=None):
        # We are training and we can do teacher forcing and batch processing
        if target is not None:
            # Initalize output
            output={'predicted_labels': torch.zeros(sum(target['tree_sizes']), self.vocab_size, device=self.device),
                    'labels': torch.zeros(sum(target['tree_sizes']), device=self.device),
                    'predicted_has_siblings': torch.zeros(sum(target['tree_sizes']),  device=self.device),
                    'has_siblings': torch.zeros(sum(target['tree_sizes']), device=self.device),
                    'predicted_is_parent': torch.zeros(sum(target['tree_sizes']), device=self.device),
                    'is_parent': torch.zeros(sum(target['tree_sizes']), device=self.device)}


            node_order = target['node_order_topdown']
            edge_order = target['edge_order_topdown']
            features = target['features']
            adj_list = target['adjacency_list']

            total_nodes = node_order.shape[0]

            # h and c states for every node in the batch for parent lstm
            h_p = torch.zeros(total_nodes, self.latent_size, device=self.device)
            c_p = torch.zeros(total_nodes, self.latent_size, device=self.device)
            
            #  h and c states for every node in the batch for sibling lstm
            h_s = torch.zeros(total_nodes, self.latent_size, device=self.device)
            c_s = torch.zeros(total_nodes, self.latent_size, device=self.device)

            for iteration in range(node_order.max() + 1):
                self.decode_train(iteration, z, h_p, c_p, h_s, c_s, node_order, edge_order, features, adj_list, output)
            

            return output
        
        else:
            trees = []
            c_parent = torch.zeros(1, self.latent_size, device=self.device)
            
            for index in range(z.shape[0]):
                trees.append(self.decode_eval((z[index].unsqueeze(0), c_parent), None))
                
            return trees

            
    def decode_train(self, iteration, z, h_p, c_p, h_s, c_s, node_order, edge_order, features, adj_list, output):
        batch_size = z.shape[0]
        
        # At iteration 0, we are at the root so we want to take the latent vector as hidden parent
        if iteration == 0:
            # Initialize to batch size (consecutive numbers s.t. the root nodes do not have siblings, see calc of siblings below)
            parent_indices = [i for i in range(batch_size)]
        else:
            # Get adjacency list of prev iteration (so parents)
            adj_list_prev = adj_list[edge_order == iteration - 1, :]
            # Get the parent indices
            parent_indices = adj_list_prev[:, 0].cpu()
            # Get the indices of the current nodes
            current_indices = adj_list_prev[:, 1]


        # Get index of first siblings of current nodes and sizes of sibling groups
        _, first_sibling_indices, sibling_group_sizes = np.unique(parent_indices, return_index=True, return_counts=True)

        # Find the largest number of siblings as we might have sibling groups of different sizes
        largest_num_siblings = max(sibling_group_sizes)

        # Iterate over sibling indices, start with first sibling, then move to next as we need the hidden state of the previous sibling
        # For the next sibling
        for sibling_index in range(largest_num_siblings):
            h_parent, c_parent, h_prev_sibling, c_prev_sibling, is_parent, has_sibling, current_nodes_indices = \
                                                                            self.get_hidden_values(iteration, adj_list, edge_order, h_p, c_p, h_s, c_s,
                                                                                                   sibling_index, first_sibling_indices,
                                                                                                   current_indices if iteration > 0 else [],
                                                                                                   node_order, parent_indices, z)
            
            h_pred = torch.tanh(self.U_parent(h_parent) + self.U_sibling(h_prev_sibling))

            # Label prediction from hidden state
            label_pred = self.label_prediction(h_pred)

            # Probability of the node having children
            p_parent = self.sigmoid(self.depth_pred(h_pred))
            # Probability of the node having successor children
            p_sibling = self.sigmoid(self.width_pred(h_pred))
                        
            # Node label prediction
            predicted_label = self.softmax(label_pred + self.offset_parent(is_parent) + self.offset_sibling(has_sibling))

            # Get true label value and its one hot encoding
            label = features[current_nodes_indices].long()
            label_onehot = F.one_hot(label, self.vocab_size).float().view(-1, self.vocab_size).to(self.device)
            
            # Compute hidden and cell values of current nodes
            h_parent, c_parent = self.lstm_parent(label_onehot, (h_parent, c_parent))
            h_sibling, c_sibling = self.lstm_sibling(label_onehot, (h_prev_sibling, c_prev_sibling))
            
            # Update the hidden and cell values matrices
            h_p[current_nodes_indices] = h_parent
            c_p[current_nodes_indices] = c_parent
            
            h_s[current_nodes_indices] = h_sibling
            c_s[current_nodes_indices] = c_sibling
                        
            # For computing loss, save output (predictions and true values)
            output['predicted_labels'][current_nodes_indices] = predicted_label
            output['labels'][current_nodes_indices] = features[current_nodes_indices].view(-1)
            output['predicted_has_siblings'][current_nodes_indices] = p_sibling.view(-1)
            output['has_siblings'][current_nodes_indices] = has_sibling.view(-1)
            output['predicted_is_parent'][current_nodes_indices] = p_parent.view(-1)
            output['is_parent'][current_nodes_indices] = is_parent.view(-1)

    def get_hidden_values(self, iteration, adj_list, edge_order, h_p, c_p, h_s, c_s, sibling_index,
                          first_sibling_indices, current_indices, node_order, parent_indices, z):
        
        batch_size = z.shape[0]
        
        # At sibling index 0, there should not be any previous siblings
        if sibling_index == 0:
            num_first_siblings = len(first_sibling_indices)
            # Initialize to hidden, cell of siblings to zero
            h_prev_sibling = torch.zeros(num_first_siblings, self.latent_size, device=self.device)
            c_prev_sibling = torch.zeros(num_first_siblings, self.latent_size, device=self.device)

            if iteration == 0:
                current_nodes_indices = torch.where(node_order == iteration)[0]
            else:
                current_nodes_indices = current_indices[first_sibling_indices]
                parent_indices_siblings = parent_indices[first_sibling_indices]

        else:           
            indices = []

            for i, first_sibling_index in enumerate(first_sibling_indices):
                if (i + 1 < len(first_sibling_indices) and sibling_index + first_sibling_index < first_sibling_indices[i + 1])\
                    or (i + 1 == len(first_sibling_indices) and sibling_index + first_sibling_index < len(current_indices)):
                    indices.append(first_sibling_index + sibling_index - 1)

            prev_siblings_indices = current_indices[indices]
            h_prev_sibling = h_s[prev_siblings_indices, :]
            c_prev_sibling = c_s[prev_siblings_indices, :]

            parent_indices_siblings = parent_indices[indices]
            current_nodes_indices = current_indices[[ind + 1 for ind in indices]]

        # Iteration 0: Root node, so there are no parents
        if iteration == 0:            
            h_parent = z
            c_parent = torch.zeros(batch_size, self.latent_size, device=self.device)    
        else:
            h_parent = h_p[parent_indices_siblings, :]
            c_parent = c_p[parent_indices_siblings, :]

        adj_list_curr = adj_list[edge_order == iteration, :]
        sib = list(first_sibling_indices) + [len(parent_indices)]
        is_parent = torch.tensor([[1.] if index in adj_list_curr[:, 0] else [0.] for index in current_nodes_indices], device=self.device)
        has_sibling = torch.tensor([[1.] if j-i -1 > sibling_index else [0.] for i,  j in zip(sib[:-1], sib[1:]) if j-i > sibling_index], device=self.device)
        
        
        return h_parent, c_parent, h_prev_sibling, c_prev_sibling, is_parent, has_sibling, current_nodes_indices
    
    
    def decode_eval(self, parent_state, sibling_state, parent_node=None):        
        h_parent, c_parent = parent_state
            
        if sibling_state is not None:
            h_prev_sibling, c_prev_sibling = sibling_state
        else:
            # Initialize to hidden, cell of siblings to zero
            h_prev_sibling = torch.zeros(1, self.latent_size, device=self.device)
            c_prev_sibling = torch.zeros(1, self.latent_size, device=self.device)
            
            
        h_pred = torch.tanh(self.U_parent(h_parent) + self.U_sibling(h_prev_sibling))
        
        # Label prediction from hidden state
        label_pred = self.label_prediction(h_pred)

        # Probability of the node having children
        p_parent = self.sigmoid(self.depth_pred(h_pred))
        # Probability of the node having successor children
        p_sibling = self.sigmoid(self.width_pred(h_pred))
        
        # Sample is_parent and has_sibling from predicted probability of parent/sibling
        is_parent = torch.distributions.bernoulli.Bernoulli(p_parent).sample()
        has_sibling = torch.distributions.bernoulli.Bernoulli(p_sibling).sample()

        # Could also simply use > 0.5 instead OR TODO BEAM SEARCH
        # is_parent = torch.tensor(1) if p_parent > 0.5 else torch.tensor(0)
        # has_sibling = torch.tensor(1) if p_sibling > 0.5 else torch.tensor(0)

        # Node label prediction
        predicted_label = self.softmax(label_pred + self.offset_parent(is_parent) + self.offset_sibling(has_sibling))
        
        # Build tree: Add node to tree
        if parent_node is None:
            node = Node(torch.argmax(predicted_label, dim=-1), parent=None)
        else:
            node = Node(torch.argmax(predicted_label, dim=-1), parent=parent_node)
                    
        # Take argmax of predicted label and transform to onehot
        predicted_label = F.one_hot(torch.argmax(predicted_label, dim=-1), self.vocab_size).float().view(-1, self.vocab_size).to(self.device)
                    
        # If we predict a next sibling
        if has_sibling:
            # Calculate next hidden sibling state
            sibling_state = self.lstm_sibling(predicted_label, (h_prev_sibling, c_prev_sibling))
            
            # Pass the same parent state, but updated sibling state
            self.decode_eval(parent_state, sibling_state, parent_node)
            
        # We set the created node as the parent node
        parent_node = node
        
        # If we predict we are a parent, continue with children
        if is_parent:
            # update parent state and parent_node of tree
            parent_state = self.lstm_parent(predicted_label, parent_state)
            
            # Pass new parent state and no sibling state as we start with the first sibling
            self.decode_eval(parent_state, None, parent_node)
        
        
        # If we are done, return the root node (which contains the entire tree)
        return parent_node
        
        
