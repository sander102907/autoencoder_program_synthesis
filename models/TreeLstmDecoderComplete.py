import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.TreeNode import Node
from utils.Sampling import Sampling
from time import time
from model_utils.modules import AddGate, PredictiveHidden, TreeTopologyPred
from model_utils.state import TreeState

# TODO make flag to combine output of LSTM cell and hidden state for predictive purposes or just take the hidden OR cell state


class TreeLstmDecoderComplete(nn.Module):
    def __init__(self, device, params, embedding_layers):
        super().__init__()

        self.device = device
        self.params = params
        self.latent_dim = params['LATENT_DIM']

        # Shared embedding layers (shared with encoder)
        self.embedding_layers = embedding_layers

        # Latent to hidden layer -> transform z from latent dim to hidden size
        self.latent2hidden = nn.Linear(self.latent_dim * 2 if params['USE_CELL_LSTM_OUTPUT'] else self.latent_dim, self.params['HIDDEN_SIZE'])

        # Doubly recurrent network layers: parent and sibling RNNs
        self.lstms_parent = nn.ModuleList([])
        self.lstms_sibling = nn.ModuleList([])

        self.pred_hidden_state = PredictiveHidden(self.latent_dim * 2 if params['USE_CELL_LSTM_OUTPUT'] else self.latent_dim)

        self.add_gate = AddGate(self.latent_dim * 2 if params['USE_CELL_LSTM_OUTPUT'] else self.latent_dim)

        self.tree_topology_pred = TreeTopologyPred(self.latent_dim * 2 if params['USE_CELL_LSTM_OUTPUT'] else self.latent_dim)

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
                    self.latent_dim * 2 if self.params['USE_CELL_LSTM_OUTPUT'] else self.latent_dim,
                    self.params['TOP_NAMES_TO_KEEP'] + self.params['NAME_ID_VOCAB_SIZE'])

                # Name token cross entropy loss
                self.label_losses[k] = nn.CrossEntropyLoss(
                    weight=self.params[f'{k}_WEIGHTS'], reduction='sum')

            elif k == 'LITERAL':
                # Adaptive softmax loss does not need prediction layer, much faster approach
                # for calculating softmax with highly imbalanced vocabs
                # cutoffs: 10: 71.1%, 11-100: 18.8%, 101-1000: 7.8%, rest: 2.2%
                self.label_losses[k] = nn.AdaptiveLogSoftmaxWithLoss(
                    self.latent_dim * 2 if self.params['USE_CELL_LSTM_OUTPUT'] else self.latent_dim,
                    self.params[f'{k}_VOCAB_SIZE'], cutoffs=[10, 100, 1000], div_value=3.0)

            else:
                # Prediction layer for RES and LITERAL tokens
                self.prediction_layers[k] = nn.Linear(
                    self.latent_dim * 2 if self.params['USE_CELL_LSTM_OUTPUT'] else self.latent_dim, 
                    self.params[f'{k}_VOCAB_SIZE'])

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
                ['PARENT', 'SIBLING', 'IS_RES']

            for loss_type in loss_types:
                individual_losses[loss_type] = 0
                accuracies[loss_type] = 0

            node_order = target['node_order_topdown']

            # Total number of nodes of all trees in the batch
            total_nodes = node_order.shape[0]

            tree_state = TreeState(target, self.params, self.device)

            # Iterate over the levels of the tree -> top down
            for iteration in range(node_order.max() + 1):
                loss = self.decode_train(iteration, z, tree_state, individual_losses, accuracies)
                total_loss += loss


            for loss_type in loss_types:
                if loss_type in ['PARENT', 'SIBLING', 'IS_RES']:
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

    def decode_train(self, iteration, z, tree_state, individual_losses, accuracies):
        loss = 0

        # At iteration 0, we are at the root
        if iteration == 0:
            hidden = self.latent2hidden(z)

            if self.params['USE_CELL_LSTM_OUTPUT']:
                h_parent, c_parent = torch.split(hidden, int(hidden.shape[-1]/2), dim=-1)
            else:
                h_parent = hidden
                c_parent = torch.zeros((z.shape[0], self.params['LATENT_DIM']), device=self.device)

            emb_label = self.embedding_layers['RES'](tree_state.features[tree_state.node_order == 0]).view(-1, self.params['EMBEDDING_DIM'])

            # Compute hidden and cell values of current nodes
            for i in range(self.params['NUM_LSTM_LAYERS']):
                if i == 0:
                    h_parent_new, c_parent_new = self.lstms_parent[i](emb_label, (h_parent, c_parent))
                else:
                    h_parent_new, c_parent_new = self.lstms_parent[i](h_parent_new, (h_parent, c_parent))

                # Update the hidden and cell values matrices
                tree_state.h_p[i][tree_state.node_order == 0] = h_parent
                tree_state.c_p[i][tree_state.node_order == 0] = c_parent

        else:
            # Get adjacency list of prev iteration (so parents)
            adj_list_prev = tree_state.adj_list[tree_state.edge_order == iteration - 1, :]
            # Get the parent indices
            parent_indices = adj_list_prev[:, 0].cpu()

            # Get index of first siblings of current nodes and sizes of sibling groups
            _, first_sibling_indices, sibling_group_sizes = np.unique(
                parent_indices, return_index=True, return_counts=True)

            # Find the largest number of siblings as we might have sibling groups of different sizes
            largest_num_siblings = max(sibling_group_sizes)

            # Iterate over sibling indices, start with first sibling, then move to next as we need the hidden state of the previous sibling
            # For the next sibling
            for sibling_index in range(largest_num_siblings):
                tree_state.update_hidden_values(iteration, sibling_index, first_sibling_indices, parent_indices)

                # print(tree_state.h_parent)

                # Combine hidden and cell states from last RNN layer
                if self.params['USE_CELL_LSTM_OUTPUT']:
                    parent_state = torch.cat([tree_state.h_parent[-1], tree_state.c_parent[-1]], dim=-1)
                    sibling_state = torch.cat([tree_state.h_prev_sibling[-1], tree_state.c_prev_sibling[-1]], dim=-1)
                else:
                    parent_state = tree_state.h_parent[-1]
                    sibling_state = tree_state.h_prev_sibling[-1]


                h_pred = self.pred_hidden_state(parent_state, sibling_state)


                # Following the Add gate of the LSTM cell, apply add gate on sibling hidden state and add to parent state
                # This way we can choose to add some sibling information in parent information
                # Think about the case when there are two func declarations, 
                # without this add gate, e.g. the function names would get exactly the same parent state                
                parent_state_updated = parent_state + self.add_gate(sibling_state)

                if self.params['USE_CELL_LSTM_OUTPUT']:
                    tree_state.h_parent[-1], tree_state.c_parent[-1] = torch.split(parent_state_updated, int(parent_state_updated.shape[-1]/2), dim=-1)
                else:
                    tree_state.h_parent[-1] = parent_state_updated

                depth_pred, width_pred, res_pred = self.tree_topology_pred(h_pred)

                # Calculate parent and sibling loss
                parent_loss = self.bce_loss(depth_pred, tree_state.is_parent) / tree_state.total_nodes
                sibling_loss = self.bce_loss(width_pred, tree_state.has_sibling) / tree_state.total_nodes

                # calculate res prediction loss
                res_loss = self.bce_loss(res_pred, tree_state.is_res) / tree_state.total_nodes

                loss += parent_loss + sibling_loss + res_loss

                individual_losses['PARENT'] += parent_loss.item()
                individual_losses['SIBLING'] += sibling_loss.item()
                individual_losses['IS_RES'] += res_loss.item()

                accuracies['PARENT'] += sum(
                    (self.sigmoid(depth_pred) >= 0.5) == (tree_state.is_parent == 1)).item()
                accuracies['SIBLING'] += sum(
                    (self.sigmoid(width_pred) >= 0.5) == (tree_state.has_sibling == 1)).item()

                accuracies['IS_RES'] += sum(
                    (self.sigmoid(res_pred) >= 0.5) == (tree_state.is_res == 1)).item()


                # Get true label values
                label = tree_state.features[tree_state.current_nodes_indices].long()

                # print(iteration, sibling_index, parent_state)

                # Iterate over possible node types and predict labels for each node type
                for k, prediction_layer in list(self.prediction_layers.items()) + [('LITERAL', _)]:
                    # Only do actual calculations when the amount of nodes for this node type > 0
                    if len(h_pred[tree_state.vocabs_mask == k]) > 0:
                        # Get label predictions
                        if k == 'LITERAL':
                            # TODO look into how we can incorporate offset parent and offset sibling here (see below)
                            # or if it even influences our loss since we work with the vocab types so maybe we can remove it altogether
                            label_pred, label_loss = self.label_losses[k](
                                h_pred[tree_state.vocabs_mask == k], label[tree_state.vocabs_mask == k].view(-1))

                            label_loss /= sum(tree_state.vocabs== k)

                            accuracies[k] += sum(self.label_losses[k].predict(h_pred[tree_state.vocabs_mask == k])
                                                == label[tree_state.vocabs_mask == k].view(-1)).item()

                        else:
                            label_pred = prediction_layer(h_pred[tree_state.vocabs_mask == k])


                            # Calculate cross entropy loss of label prediction
                            label_loss = self.label_losses[k]((label_pred
                                                            # + self.offset_parent(is_parent[vocabs_mask == k])
                                                            # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                            ), label[tree_state.vocabs_mask == k].view(-1)) / sum(tree_state.vocabs == k)


                            accuracies[k] += sum(torch.argmax(self.softmax(label_pred
                                                                        # + self.offset_parent(is_parent[vocabs_mask == k])
                                                                        # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                                        ), dim=-1) == label[tree_state.vocabs_mask == k].view(-1)).item()
                            
                            pred_labels = torch.argmax(self.softmax(label_pred), dim=-1)
                            print(iteration, sibling_index, pred_labels[pred_labels != label[tree_state.vocabs_mask == k].view(-1)].tolist(), pred_labels.tolist(), label[tree_state.vocabs_mask == k].view(-1).tolist()) #, self.softmax(label_pred).topk(3, sorted=True)[1], self.softmax(label_pred).topk(3, sorted=True)[0], h_pred[:, :5],)
                        loss += label_loss #* max(-100 * iteration + 200, 1)

                        individual_losses[k] += label_loss.item()

                        # Calculate embedding of true label -> teacher forcing
                        if 'RES' in k or not self.params['INDIV_LAYERS_VOCABS']:
                            embedding_dim = self.params['EMBEDDING_DIM']
                        else:
                            embedding_dim = self.params['LEAF_EMBEDDING_DIM']

                        emb_label = self.embedding_layers[k](
                            label[tree_state.vocabs_mask == k]).view(-1, embedding_dim)

                        if 'RES' in k:
                            for i in range(self.params['NUM_LSTM_LAYERS']):
                                # Compute hidden and cell values of current nodes
                                if i == 0:
                                    h_parent_new, c_parent_new = self.lstms_parent[i](
                                        emb_label,
                                        (tree_state.h_parent[i][tree_state.vocabs_mask == k],
                                         tree_state.c_parent[i][tree_state.vocabs_mask == k])
                                    )

                                    h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                        emb_label,
                                        (tree_state.h_prev_sibling[i][tree_state.vocabs_mask == k],
                                         tree_state.c_prev_sibling[i][tree_state.vocabs_mask == k])
                                    )
                                else:
                                    h_parent_new, c_parent_new = self.lstms_parent[i](
                                        h_parent_new, 
                                        (tree_state.h_parent[i][tree_state.vocabs_mask == k], 
                                         tree_state.c_parent[i][tree_state.vocabs_mask == k])
                                    )

                                    h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                        h_sibling_new, 
                                        (tree_state.h_prev_sibling[i][tree_state.vocabs_mask == k], 
                                         tree_state.c_prev_sibling[i][tree_state.vocabs_mask == k])
                                    )

                                # Update the hidden and cell values matrices
                                tree_state.h_p[i][tree_state.current_nodes_indices[tree_state.vocabs_mask == k]] = h_parent_new
                                tree_state.c_p[i][tree_state.current_nodes_indices[tree_state.vocabs_mask == k]] = c_parent_new

                                tree_state.h_s[i][tree_state.current_nodes_indices[tree_state.vocabs_mask == k]] = h_sibling_new
                                tree_state.c_s[i][tree_state.current_nodes_indices[tree_state.vocabs_mask == k]] = c_sibling_new

                        else:
                            # Compute hidden and cell values of current nodes for previous siblings only (since we are not parents in the leafs)
                            if self.params['INDIV_LAYERS_VOCABS']:
                                h, c = self.leaf_lstms_sibling[k](
                                    emb_label,
                                    (tree_state.h_prev_sibling[tree_state.vocabs_mask == k],
                                     tree_state.c_prev_sibling[tree_state.vocabs_mask == k])
                                )
                            else:
                                for i in range(self.params['NUM_LSTM_LAYERS']):
                                    if i == 0:
                                        h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                            emb_label, 
                                            (tree_state.h_prev_sibling[i][tree_state.vocabs_mask == k], 
                                             tree_state.c_prev_sibling[i][tree_state.vocabs_mask == k])
                                        )

                                    else:
                                        h_sibling_new, c_sibling_new = self.lstms_sibling[i](
                                            h_sibling_new, 
                                            (tree_state.h_prev_sibling[i][tree_state.vocabs_mask == k], 
                                             tree_state.c_prev_sibling[i][tree_state.vocabs_mask == k])
                                        )

                                    # Update the hidden and cell values matrices
                                    tree_state.h_s[i][tree_state.current_nodes_indices[tree_state.vocabs_mask == k]] = h_sibling_new
                                    tree_state.c_s[i][tree_state.current_nodes_indices[tree_state.vocabs_mask == k]] = c_sibling_new

        return loss


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

            h_pred = self.pred_hidden_state(parent_state_combined, sibling_state_combined)

            # sibling add gate to add to parent state           
            parent_state_updated = parent_state_combined + self.add_gate(sibling_state_combined)

            parent_state_updated = torch.split(parent_state_updated, int(parent_state_updated.shape[-1]/2), dim=-1)


            depth_pred, width_pred, res_pred = self.tree_topology_pred(h_pred)

            # Probability of the node having children
            p_parent = self.sigmoid(depth_pred)
            # Probability of the node having successor children
            p_sibling = self.sigmoid(width_pred)

            # Probability of the node being a reserved c++ token
            p_res = self.sigmoid(res_pred)

            # Sample is_parent and has_sibling from predicted probability of parent/sibling
            is_parent = True if p_parent >= 0.5 else False
            has_sibling = True if p_sibling >= 0.5 else False
            # is_parent = torch.distributions.bernoulli.Bernoulli(p_parent).sample()
            # has_sibling = torch.distributions.bernoulli.Bernoulli(
            #     p_sibling).sample()

            is_res = True if p_res >= 0.5 else False


            if is_res:
                node_type = 'RES'
            elif 'LITERAL' in idx_to_label[parent_node.token]:
                node_type = 'LITERAL'
            elif 'TYPE' == idx_to_label[parent_node.token]:
                node_type = 'TYPE'
            else:
                node_type = 'NAME'

            # Node label prediction
            if node_type == 'LITERAL':
                label_logits = self.label_losses[node_type].log_prob(h_pred)
            else:
                label_logits = self.prediction_layers[node_type](h_pred)

            predicted_label = Sampling.sample(label_logits.view(-1), temperature=0.9, top_k=40, top_p=0.9)


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
