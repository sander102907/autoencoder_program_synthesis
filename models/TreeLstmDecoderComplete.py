import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.TreeNode import Node
from time import time


class TreeLstmDecoderComplete(nn.Module):
    def __init__(self, device, params, embedding_layers):
        super().__init__()

        self.device = device
        self.params = params
        self.latent_dim = params['LATENT_DIM']

        # Shared embedding layers (shared with encoder)
        self.embedding_layers = embedding_layers

        # Doubly recurrent network layers: parent and sibling RNNs
        self.lstms_parent = nn.ModuleList([])
        self.lstms_sibling = nn.ModuleList([])

        # Learnable parameter U_parent -> for calculating h_pred: U_parent * hidden_parent
        self.U_parent = nn.Linear(
            params['LATENT_DIM'] * 2, params['LATENT_DIM'] * 2, bias=True)

        # For topology prediction, we predict whether there are children -> depth_pred * h_pred
        self.depth_pred = nn.Linear(params['LATENT_DIM'] * 2, 1)

        # Learnable parameter U_sibling -> for calculating h_pred: U_sibling * hidden_sibling
        self.U_sibling = nn.Linear(
            params['LATENT_DIM'] * 2, params['LATENT_DIM'] * 2, bias=True)

        # For topology prediction, we predict whether there are successor siblings -> width_pred * h_pred
        self.width_pred = nn.Linear(params['LATENT_DIM'] * 2, 1)


        self.W_s_mul = nn.Linear(params['LATENT_DIM'] * 2, params['LATENT_DIM'] * 2, bias=True)
        self.W_s_add = nn.Linear(params['LATENT_DIM'] * 2, params['LATENT_DIM'] * 2, bias=True)


        # Learnable weights to incorporate topology information in label prediction
        self.offset_parent = nn.Linear(1, 1, bias=True)
        self.offset_sibling = nn.Linear(1, 1, bias=True)

        # Leaf lstms, if we want individual RNNs for leaf nodes
        self.leaf_lstms_sibling = nn.ModuleDict({})

        # Prediction layers for each vocab
        self.prediction_layers = nn.ModuleDict({})

        # Binary cross entropy loss for computing loss of topology
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # Loss functions for labels for each vocab so we can specify weights if needed
        self.label_losses = nn.ModuleDict({})


        # For evaluation/testing/generating:

        # sigmoid to predict topology of the tree (width and depth)
        self.sigmoid = nn.Sigmoid()

        # softmax to get probability distribution for labels of nodes
        self.softmax = nn.LogSoftmax(dim=-1)


        # Initialize layers
        self.init_layers()


    def init_layers(self):
        # Initialize RNNs with num layers
        for i in range(self.params['NUM_LSTM_LAYERS']):
            if i == 0:
                self.lstms_parent.append(nn.LSTMCell(self.params['EMBEDDING_DIM'], self.params['LATENT_DIM']))
                self.lstms_sibling.append(nn.LSTMCell(self.params['EMBEDDING_DIM'], self.params['LATENT_DIM']))
            else:
                self.lstms_parent.append(nn.LSTMCell(self.params['LATENT_DIM'], self.params['LATENT_DIM']))
                self.lstms_sibling.append(nn.LSTMCell(self.params['LATENT_DIM'], self.params['LATENT_DIM']))


        # Initialize prediction layers, loss functions, leaf sibling LSTMs
        for k, embedding_layer in self.embedding_layers.items():
            if not 'RES' in k and self.params['INDIV_LAYERS_VOCABS']:
                # Individual RNNs for leaf tokens
                self.leaf_lstms_sibling[k] = nn.LSTMCell(
                    self.params['LEAF_EMBEDDING_DIM'], self.params['HIDDEN_SIZE'])

            if k == 'NAME':
                # For name tokens we predict reusable IDs of predefined size instead of actual tokens
                self.prediction_layers[k] = nn.Linear(
                    self.params['LATENT_DIM'] * 2, self.params['TOP_NAMES_TO_KEEP'] + self.params['NAME_ID_VOCAB_SIZE'])

                # Name token cross entropy loss
                self.label_losses[k] = nn.CrossEntropyLoss(
                    weight=self.params[f'{k}_WEIGHTS'], reduction='sum')

            elif k == 'LITERAL':
                # Adaptive softmax loss does not need prediction layer, much faster approach
                # for calculating softmax with highly imbalanced vocabs
                # cutoffs: 10: 71.1%, 11-100: 18.8%, 101-1000: 7.8%, rest: 2.2%
                self.label_losses[k] = nn.AdaptiveLogSoftmaxWithLoss(
                    self.params['LATENT_DIM'] * 2, self.params[f'{k}_VOCAB_SIZE'], cutoffs=[10, 100, 1000], div_value=3.0)

            else:
                # Prediction layer for RES and LITERAL tokens
                self.prediction_layers[k] = nn.Linear(
                    self.params['LATENT_DIM'] * 2, self.params[f'{k}_VOCAB_SIZE'])

                # RES and LITERAL cross entropy loss
                self.label_losses[k] = nn.CrossEntropyLoss(
                    weight=self.params[f'{k}_WEIGHTS'], reduction='sum')


    def forward(self, z, target=None, idx_to_label=None, nameid_to_placeholderid=None):
        """
        @param z: (batch_size, LATENT_DIM * 2) -> latent vector which is 2 * LATENT DIM as it contains
                                                  latent vector for both hidden and cell state of LSTM

        @param target: dictionary containing tree information -> node_order_topdown, edge_order_topdown,
                                                  features, adjacency_list and vocabs

        @param idx_to_label: dictionary containing mapping from ids to labels, needed for evaluation
        """

        # We are training and we can do teacher forcing and batch processing
        if target is not None:
            # Keep track of the loss
            total_loss = 0
            individual_losses = {}
            accuracies = {}
            loss_types = list(self.embedding_layers.keys()) + \
                ['PARENT', 'SIBLING']

            for loss_type in loss_types:
                individual_losses[loss_type] = 0
                accuracies[loss_type] = 0

            node_order = target['node_order_topdown']

            # Total number of nodes of all trees in the batch
            total_nodes = node_order.shape[0]

            h_p = []
            c_p = []

            h_s = []
            c_s = []

            for _ in range(self.params['NUM_LSTM_LAYERS']):
                # h and c states for every node in the batch for parent lstm
                h_p.append(torch.zeros(total_nodes, self.latent_dim, device=self.device))
                c_p.append(torch.zeros(total_nodes, self.latent_dim, device=self.device))

                # h and c states for every node in the batch for sibling lstm
                h_s.append(torch.zeros(total_nodes, self.latent_dim, device=self.device))
                c_s.append(torch.zeros(total_nodes, self.latent_dim, device=self.device))

            # Iterate over the levels of the tree -> top down
            for iteration in range(node_order.max() + 1):
                loss = self.decode_train(
                    iteration, z, h_p, c_p, h_s, c_s, target, individual_losses, accuracies)
                total_loss += loss

            for loss_type in loss_types:
                if loss_type in ['PARENT', 'SIBLING']:
                    accuracies[loss_type] = accuracies[loss_type] / (total_nodes - z.shape[0])
                    # print(loss_type, accuracies[loss_type])
                else:
                    if loss_type == 'RES':
                        # Correct for root node -> is not predicted
                        accuracies[loss_type] = accuracies[loss_type] / (sum(target['vocabs'] == loss_type) - z.shape[0])
                    else:
                        accuracies[loss_type] = accuracies[loss_type] / sum(target['vocabs'] == loss_type)

            return total_loss, individual_losses, accuracies

         # We are evaluating and we cannot use training forcing and we generate tree by tree
        elif idx_to_label is not None:
            trees = []
            label_to_idx = {v:k for k,v in idx_to_label.items()}

            for index in range(z.shape[0]):
                if nameid_to_placeholderid is not None:
                    placeholderid_to_nameid = {
                        v: k for k, v in nameid_to_placeholderid[index].items()
                    }
                else:
                    placeholderid_to_nameid = None

                h_parent, c_parent = torch.split(
                    z[index], int(len(z[index])/2)
                )

                parent_state = [(h_parent.unsqueeze(0), c_parent.unsqueeze(0))] + [None for _ in range(self.params['NUM_LSTM_LAYERS'] - 1)]
                sibling_state = [None for _ in range(self.params['NUM_LSTM_LAYERS'])]

                trees.append(self.decode_eval(parent_state, sibling_state, idx_to_label, label_to_idx, placeholderid_to_nameid))

            return trees

    def decode_train(self, iteration, z, h_p, c_p, h_s, c_s, target, individual_losses, accuracies):
        loss = 0

        # Get needed input data
        node_order = target['node_order_topdown']
        edge_order = target['edge_order_topdown']
        features = target['features']
        adj_list = target['adjacency_list']
        vocabs = target['vocabs']
        total_nodes = node_order.shape[0]

        # At iteration 0, we are at the root
        if iteration == 0:
            h_parent, c_parent = torch.split(z, int(z.shape[-1]/2), dim=-1)
            # print(z[:, :5])
            emb_label = self.embedding_layers['RES'](features[node_order == 0]).view(-1, self.params['EMBEDDING_DIM'])

            # Compute hidden and cell values of current nodes
            for i in range(self.params['NUM_LSTM_LAYERS']):
                if i == 0:
                    h_parent_new, c_parent_new = self.lstms_parent[i](emb_label, (h_parent, c_parent))
                else:
                    h_parent_new, c_parent_new = self.lstms_parent[i](h_parent_new, None)

                # Update the hidden and cell values matrices
                h_p[i][node_order == 0] = h_parent
                c_p[i][node_order == 0] = c_parent

        else:
            # Get adjacency list of prev iteration (so parents)
            adj_list_prev = adj_list[edge_order == iteration - 1, :]
            # Get the parent indices
            parent_indices = adj_list_prev[:, 0].cpu()
            # Get the indices of the current nodes
            current_indices = adj_list_prev[:, 1]

            # Get index of first siblings of current nodes and sizes of sibling groups
            _, first_sibling_indices, sibling_group_sizes = np.unique(
                parent_indices, return_index=True, return_counts=True)

            # Find the largest number of siblings as we might have sibling groups of different sizes
            largest_num_siblings = max(sibling_group_sizes)

            # Iterate over sibling indices, start with first sibling, then move to next as we need the hidden state of the previous sibling
            # For the next sibling
            for sibling_index in range(largest_num_siblings):
                h_parent, c_parent, h_prev_sibling, c_prev_sibling, is_parent, has_sibling, current_nodes_indices, vocabs_mask = \
                    self.get_hidden_values(iteration, adj_list, edge_order, h_p, c_p, h_s, c_s,
                                        sibling_index, first_sibling_indices,
                                        current_indices if iteration > 0 else [],
                                        node_order, parent_indices, vocabs)

                # Combine hidden and cell states
                parent_state = torch.cat([h_parent[-1], c_parent[-1]], dim=-1)
                sibling_state = torch.cat([h_prev_sibling[-1], c_prev_sibling[-1]], dim=-1)

                h_pred = torch.tanh(self.U_parent(parent_state) +
                                    self.U_sibling(sibling_state))


                # Following the Add gate of the LSTM cell, apply add gate on sibling hidden state and add to parent state
                # This way we can choose to add some sibling information in parent information
                # Think about the case when there are two func declarations, 
                # without this add gate, e.g. the function names would get exactly the same parent state
                sibling_mul_state = self.sigmoid(self.W_s_mul(sibling_state))
                sibling_add_state = torch.tanh(self.W_s_add(sibling_state))
                
                parent_state_updated = parent_state + sibling_mul_state * sibling_add_state

                h_parent[-1], c_parent[-1] = torch.split(parent_state_updated, int(parent_state_updated.shape[-1]/2), dim=-1)

                # Calculate parent and sibling loss
                parent_loss = self.bce_loss(
                    self.depth_pred(h_pred), is_parent) / total_nodes
                sibling_loss = self.bce_loss(
                    self.width_pred(h_pred), has_sibling) / total_nodes

                loss += parent_loss + sibling_loss

                individual_losses['PARENT'] += parent_loss.item()
                individual_losses['SIBLING'] += sibling_loss.item()

                accuracies['PARENT'] += sum(
                    (self.sigmoid(self.depth_pred(h_pred)) >= 0.5) == (is_parent == 1)).item()
                accuracies['SIBLING'] += sum(
                    (self.sigmoid(self.width_pred(h_pred)) >= 0.5) == (has_sibling == 1)).item()


                # Get true label values
                label = features[current_nodes_indices].long()

                # print(label[label == 21], is_parent[torch.where(label == 21)], current_nodes_indices[torch.where(label == 21)[0]])
                # Iterate over possible node types and predict labels for each node type
                for k, prediction_layer in list(self.prediction_layers.items()) + [('LITERAL', _)]:
                    # Only do actual calculations when the amount of nodes for this node type > 0
                    if len(h_pred[vocabs_mask == k]) > 0:
                        # Get label predictions
                        if k == 'LITERAL':
                            # TODO look into how we can incorporate offset parent and offset sibling here (see below)
                            # or if it even influences our loss since we work with the vocab types so maybe we can remove it altogether
                            label_pred, label_loss = self.label_losses[k](
                                h_pred[vocabs_mask == k], label[vocabs_mask == k].view(-1))

                            label_loss /= sum(target['vocabs'] == k)

                            accuracies[k] += sum(self.label_losses[k].predict(h_pred[vocabs_mask == k])
                                                == label[vocabs_mask == k].view(-1)).item()

                        else:
                            label_pred = prediction_layer(h_pred[vocabs_mask == k])


                            # Calculate cross entropy loss of label prediction
                            label_loss = self.label_losses[k]((label_pred
                                                            # + self.offset_parent(is_parent[vocabs_mask == k])
                                                            # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                            ), label[vocabs_mask == k].view(-1)) / sum(target['vocabs'] == k)


                            accuracies[k] += sum(torch.argmax(self.softmax(label_pred
                                                                        # + self.offset_parent(is_parent[vocabs_mask == k])
                                                                        # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                                        ), dim=-1) == label[vocabs_mask == k].view(-1)).item()

                            # pred_labels = torch.argmax(self.softmax(label_pred), dim=-1)
                            # print(sibling_index, iteration, pred_labels[pred_labels != label[vocabs_mask == k].view(-1)].tolist(), pred_labels.tolist(), label[vocabs_mask == k].view(-1).tolist()) #, self.softmax(label_pred).topk(3, sorted=True)[1], self.softmax(label_pred).topk(3, sorted=True)[0], h_pred[:, :5],)
                        loss += label_loss

                        individual_losses[k] += label_loss.item()

                        # Calculate embedding of true label -> teacher forcing
                        if 'RES' in k or not self.params['INDIV_LAYERS_VOCABS']:
                            embedding_dim = self.params['EMBEDDING_DIM']
                        else:
                            embedding_dim = self.params['LEAF_EMBEDDING_DIM']

                        emb_label = self.embedding_layers[k](
                            label[vocabs_mask == k]).view(-1, embedding_dim)

                        if 'RES' in k:
                            for i in range(self.params['NUM_LSTM_LAYERS']):
                                # Compute hidden and cell values of current nodes
                                if i == 0:
                                    h_parent_new, c_parent_new = self.lstms_parent[i](
                                        emb_label,
                                        (h_parent[i][vocabs_mask == k],
                                         c_parent[i][vocabs_mask == k])
                                    )

                                    h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                        emb_label,
                                        (h_prev_sibling[i][vocabs_mask == k],
                                        c_prev_sibling[i][vocabs_mask == k])
                                    )
                                else:
                                    h_parent_new, c_parent_new = self.lstms_parent[i](
                                        h_parent_new, 
                                        (h_parent[i][vocabs_mask == k], 
                                        c_parent[i][vocabs_mask == k])
                                    )

                                    h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                        h_sibling_new, 
                                        (h_prev_sibling[i][vocabs_mask == k], 
                                        c_prev_sibling[i][vocabs_mask == k])
                                    )

                                # Update the hidden and cell values matrices
                                h_p[i][current_nodes_indices[vocabs_mask == k]] = h_parent_new
                                c_p[i][current_nodes_indices[vocabs_mask == k]] = c_parent_new

                                h_s[i][current_nodes_indices[vocabs_mask == k]] = h_sibling_new
                                c_s[i][current_nodes_indices[vocabs_mask == k]] = c_sibling_new

                        else:
                            # Compute hidden and cell values of current nodes for previous siblings only (since we are not parents in the leafs)
                            if self.params['INDIV_LAYERS_VOCABS']:
                                h, c = self.leaf_lstms_sibling[k](
                                    emb_label,
                                    (h_prev_sibling[vocabs_mask == k],
                                    c_prev_sibling[vocabs_mask == k])
                                )
                            else:
                                for i in range(self.params['NUM_LSTM_LAYERS']):
                                    if i == 0:
                                        h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                            emb_label, 
                                            (h_prev_sibling[i][vocabs_mask == k], 
                                            c_prev_sibling[i][vocabs_mask == k])
                                        )

                                    else:
                                        h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                            h_sibling_new, 
                                            (h_prev_sibling[i][vocabs_mask == k], 
                                            c_prev_sibling[i][vocabs_mask == k])
                                        )

                                    # Update the hidden and cell values matrices
                                    h_s[i][current_nodes_indices[vocabs_mask == k]] = h_sibling_new
                                    c_s[i][current_nodes_indices[vocabs_mask == k]] = c_sibling_new
                                # h, c = self.lstm_sibling(
                                #     emb_label, (h_prev_sibling[vocabs_mask == k], c_prev_sibling[vocabs_mask == k]))

                            # Update hidden and cell values matrices for siblings only (leafs cannot be parents)
                            # h_s[current_nodes_indices[vocabs_mask == k]] = h
                            # c_s[current_nodes_indices[vocabs_mask == k]] = c

        return loss

    def get_hidden_values(self, iteration, adj_list, edge_order, h_p, c_p, h_s, c_s, sibling_index,
                          first_sibling_indices, current_indices, node_order, parent_indices, vocabs):
        # At sibling index 0, there should not be any previous siblings
        if sibling_index == 0:
            num_first_siblings = len(first_sibling_indices)
            # Initialize to hidden, cell of siblings to zero
            h_prev_sibling = []
            c_prev_sibling = []
            for i in range(self.params['NUM_LSTM_LAYERS']):
                h_prev_sibling.append(torch.zeros(
                    num_first_siblings, self.latent_dim, device=self.device))
                c_prev_sibling.append(torch.zeros(
                    num_first_siblings, self.latent_dim, device=self.device))

            current_nodes_indices = current_indices[first_sibling_indices]
            parent_indices_siblings = parent_indices[first_sibling_indices]

        else:
            indices = []

            for i, first_sibling_index in enumerate(first_sibling_indices):
                if (i + 1 < len(first_sibling_indices) and sibling_index + first_sibling_index < first_sibling_indices[i + 1])\
                        or (i + 1 == len(first_sibling_indices) and sibling_index + first_sibling_index < len(current_indices)):
                    indices.append(first_sibling_index + sibling_index - 1)

            prev_siblings_indices = current_indices[indices]

            h_prev_sibling = []
            c_prev_sibling = []

            for i in range(self.params['NUM_LSTM_LAYERS']):
                h_prev_sibling.append(h_s[i][prev_siblings_indices, :])
                c_prev_sibling.append(c_s[i][prev_siblings_indices, :])

            parent_indices_siblings = parent_indices[indices]
            current_nodes_indices = current_indices[[
                ind + 1 for ind in indices]]

        h_parent = []
        c_parent = []

        for i in range(self.params['NUM_LSTM_LAYERS']):
            h_parent.append(h_p[i][parent_indices_siblings, :])
            c_parent.append(c_p[i][parent_indices_siblings, :])

        adj_list_curr = adj_list[edge_order == iteration, :]
        sib = list(first_sibling_indices) + [len(parent_indices)]
        vocabs_mask = np.atleast_1d(vocabs[current_nodes_indices.cpu()])
        is_parent = torch.tensor([[1.] if index in adj_list_curr[:, 0] else [
                                 0.] for index in current_nodes_indices], device=self.device)
        has_sibling = torch.tensor([[1.] if j-i - 1 > sibling_index else [0.]
                                    for i,  j in zip(sib[:-1], sib[1:]) if j-i > sibling_index], device=self.device)


        return h_parent, c_parent, h_prev_sibling, c_prev_sibling, is_parent, has_sibling, current_nodes_indices, vocabs_mask


     def decode_eval(self, parent_state, sibling_state, idx_to_label, label_to_idx, placeholderid_to_nameid, parent_node=None):
        # If we are at the root
        if parent_node is None:
            # Get the root node ID
            root_id = label_to_idx['root']

            # Create root node of the tree
            parent_node = Node(root_id, is_reserved=True, parent=None)

            # Update the parent state
            emb_label = self.embedding_layers['RES'](torch.tensor([root_id], device=self.device)).view(-1, self.params['EMBEDDING_DIM'])

            for i in range(self.params['NUM_LSTM_LAYERS']):
                if i == 0:
                    parent_state[i] = self.lstms_parent[i](emb_label, parent_state[i])
                else:
                    parent_state[i] = self.lstms_parent[i](parent_state[i-1][0], None)

            # Pass new parent state and no sibling state as we start with the first sibling
            sibling_state = [None for _ in range(self.params['NUM_LSTM_LAYERS'])]

            self.decode_eval(parent_state, sibling_state, idx_to_label, label_to_idx,
                             placeholderid_to_nameid, parent_node)

        else:
            h_parent, c_parent = parent_state[-1]

            if not None in sibling_state:
                h_prev_sibling, c_prev_sibling = sibling_state[-1]
            else:
                # Initialize to hidden, cell of siblings to zero
                h_prev_sibling = torch.zeros(
                    1, self.latent_dim, device=self.device)

                c_prev_sibling = torch.zeros(
                    1, self.latent_dim, device=self.device)

            parent_state_combined = torch.cat([h_parent, c_parent], dim=-1)
            sibling_state_combined = torch.cat([h_prev_sibling, c_prev_sibling], dim=-1)

            h_pred = torch.tanh(self.U_parent(parent_state_combined) +
                                self.U_sibling(sibling_state_combined))


            # sibling add gate to add to parent state
            sibling_mul_state = self.sigmoid(self.W_s_mul(sibling_state_combined))
            sibling_add_state = torch.tanh(self.W_s_add(sibling_state_combined))
            
            parent_state_updated = parent_state_combined + sibling_mul_state * sibling_add_state

            parent_state_updated = torch.split(parent_state_updated, int(parent_state_updated.shape[-1]/2), dim=-1)


            # Probability of the node having children
            p_parent = self.sigmoid(self.depth_pred(h_pred))
            # Probability of the node having successor children
            p_sibling = self.sigmoid(self.width_pred(h_pred))

            # Sample is_parent and has_sibling from predicted probability of parent/sibling
            # is_parent = True if p_parent >= 0.5 else False
            # has_sibling = True if p_sibling >= 0.5 else False
            is_parent = torch.distributions.bernoulli.Bernoulli(p_parent).sample()
            has_sibling = torch.distributions.bernoulli.Bernoulli(
                p_sibling).sample()


            # TODO Look into changing AST parser such that ACESS SPECIFIER child, such as public, is not a terminal anymore or not a reserved node
            # As well as ARGUMENTS that have NO children, and RETURN statement that has no children, these all do not get seen as reserved keywords here.
            if is_parent or 'ACCESS_SPECIFIER' in idx_to_label[parent_node.token] or 'COMPOUND_STMT' in idx_to_label[parent_node.token] or 'CALL_EXPR' in idx_to_label[parent_node.token]:
                node_type = 'RES'
            elif 'LITERAL' in idx_to_label[parent_node.token]:
                node_type = 'LITERAL'
            elif 'TYPE' == idx_to_label[parent_node.token]:
                node_type = 'TYPE'
            else:
                node_type = 'NAME'

            # Node label prediction
            if node_type == 'LITERAL':
                predicted_label = self.label_losses[node_type].log_prob(h_pred)
            else:
                label_pred = self.prediction_layers[node_type](h_pred)
                predicted_label = self.softmax(
                    label_pred 
                    # + self.offset_parent(is_parent) + self.offset_sibling(has_sibling)
                    )

            # TODO beam search
            predicted_label = torch.distributions.categorical.Categorical(torch.exp(predicted_label)).sample()
            # topk_log_prob, topk_indexes = predicted_label.topk(3, sorted=True)
            # predicted_label = torch.argmax(predicted_label, dim=-1)
            # print(parent_node.token, topk_indexes, topk_log_prob, p_parent)


            # Build tree: Add node to tree
            if node_type == 'NAME':
                if placeholderid_to_nameid is not None and predicted_label.item() in placeholderid_to_nameid:
                    node = Node(placeholderid_to_nameid[predicted_label.item(
                    )], is_reserved=False, parent=parent_node)
                else:
                    node = Node(f'NAME_{predicted_label.item()}', is_reserved=False, parent=parent_node)
            else:
                node = Node(predicted_label.item(
                ), is_reserved=True if node_type == 'RES' else False, parent=parent_node)


            if node_type == 'RES' or not self.params['INDIV_LAYERS_VOCABS']:
                emb_label = self.embedding_layers[node_type](
                    predicted_label).view(-1, self.params['EMBEDDING_DIM'])
            else:
                emb_label = self.embedding_layers[node_type](
                    predicted_label).view(-1, self.params['LEAF_EMBEDDING_DIM'])

            # If we predict a next sibling
            if has_sibling:
                if is_parent or not self.params['INDIV_LAYERS_VOCABS']:
                    # Calculate next hidden sibling state
                    for i in range(self.params['NUM_LSTM_LAYERS']):
                        if i == 0:
                            sibling_state[i] = self.lstms_sibling[i](emb_label, sibling_state[i])
                        else:
                            sibling_state[i] = self.lstms_sibling[i](sibling_state[i-1][0], sibling_state[i])
                else:
                    # Calculate next hidden sibling state
                    sibling_state = self.leaf_lstms_sibling[node_type](
                        emb_label, sibling_state)

                # Pass the same parent state, but updated sibling state
                self.decode_eval(parent_state, sibling_state, idx_to_label, label_to_idx,
                                placeholderid_to_nameid, parent_node)

            # We set the created node as the parent node
            parent_node = node

            # If we predict we are a parent, continue with children
            if is_parent:
                parent_state[-1] = parent_state_updated
                # update parent state and parent_node of tree
                for i in range(self.params['NUM_LSTM_LAYERS']):
                        if i == 0:
                            parent_state[i] = self.lstms_parent[i](emb_label, parent_state[i])
                        else:
                            parent_state[i] = self.lstms_parent[i](parent_state[i-1][0], parent_state[i])

                # Pass new parent state and no sibling state as we start with the first sibling
                sibling_state = [None for _ in range(self.params['NUM_LSTM_LAYERS'])]
                self.decode_eval(parent_state, sibling_state, idx_to_label, label_to_idx,
                                placeholderid_to_nameid, parent_node)

        # If we are done, return the root node (which contains the entire tree)
        return parent_node
