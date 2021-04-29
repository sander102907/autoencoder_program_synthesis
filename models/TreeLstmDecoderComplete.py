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
        self.embedding_layers = embedding_layers
        self.leaf_lstms_sibling = nn.ModuleDict({})
        self.prediction_layers = nn.ModuleDict({})
        self.label_losses = nn.ModuleDict({})

        self.lstm_parent = nn.LSTMCell(
            params['EMBEDDING_DIM'], params['LATENT_DIM'])
        self.U_parent = nn.Linear(
            params['LATENT_DIM'], params['LATENT_DIM'], bias=False)
        self.depth_pred = nn.Linear(params['LATENT_DIM'], 1)

        self.lstm_sibling = nn.LSTMCell(
            params['EMBEDDING_DIM'], params['LATENT_DIM'])
        self.U_sibling = nn.Linear(
            params['LATENT_DIM'], params['LATENT_DIM'], bias=False)
        self.width_pred = nn.Linear(params['LATENT_DIM'], 1)

        # Leaf lstms and prediction layers
        for k, embedding_layer in embedding_layers.items():
            if not 'RES' in k and params['INDIV_LAYERS_VOCABS']:
                self.leaf_lstms_sibling[k] = nn.LSTMCell(
                    params['LEAF_EMBEDDING_DIM'], params['HIDDEN_SIZE'])

            if k == 'NAME':
                self.prediction_layers[k] = nn.Linear(
                    params['LATENT_DIM'], params['NAME_ID_VOCAB_SIZE'])
                self.label_losses[k] = nn.CrossEntropyLoss(
                    weight=params[f'{k}_WEIGHTS'], reduction='sum')

            elif k == 'LITERAL':
                # Adaptive softmax loss does not need prediction layer, much faster approach for calculating softmax with highly imbalanced
                # vocab. cutoffs: 10: 71.1%, 11-100: 18.8%, 101-1000: 7.8%, rest: 2.2%
                self.label_losses[k] = nn.AdaptiveLogSoftmaxWithLoss(
                    params['LATENT_DIM'], params[f'{k}_VOCAB_SIZE'], cutoffs=[10, 100, 1000])

            else:
                self.prediction_layers[k] = nn.Linear(
                    params['LATENT_DIM'], params[f'{k}_VOCAB_SIZE'])
                self.label_losses[k] = nn.CrossEntropyLoss(
                    weight=params[f'{k}_WEIGHTS'], reduction='sum')

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=-1)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.offset_parent = nn.Linear(1, 1, bias=False)
        self.offset_sibling = nn.Linear(1, 1, bias=False)

    def forward(self, z, target=None, idx_to_label=None, nameid_to_placeholderid=None):
        """
        @param z: (batch_size, LATENT_DIM * 2) -> latent vector which is 2 * LATENT DIM as it contains latent vector for both hidden and cell state of LSTM
        @param target: dictionary containing tree information -> node_order_topdown, edge_order_topdown, features, adjacency_list and vocabs
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

            # h and c states for every node in the batch for parent lstm
            h_p = torch.zeros(total_nodes, self.latent_dim, device=self.device)
            c_p = torch.zeros(total_nodes, self.latent_dim, device=self.device)

            # h and c states for every node in the batch for sibling lstm
            h_s = torch.zeros(total_nodes, self.latent_dim, device=self.device)
            c_s = torch.zeros(total_nodes, self.latent_dim, device=self.device)

            # Iterate over the levels of the tree -> top down
            for iteration in range(node_order.max() + 1):
                loss = self.decode_train(
                    iteration, z, h_p, c_p, h_s, c_s, target, individual_losses, accuracies)
                total_loss += loss

            for loss_type in loss_types:
                if loss_type in ['PARENT', 'SIBLING']:
                    accuracies[loss_type] = (
                        accuracies[loss_type] / total_nodes).item()
                else:
                    accuracies[loss_type] = (
                        accuracies[loss_type] / sum(target['vocabs'] == loss_type)).item()

            return total_loss, individual_losses, accuracies

        # We are evaluating and we cannot use training forcing and we generate tree by tree
        elif idx_to_label is not None:
            trees = []
            pred_features = []

            for index in range(z.shape[0]):
                placeholderid_to_nameid = {
                    v: k for k, v in nameid_to_placeholderid[index].items()
                }

                h_parent, c_parent = torch.split(
                    z[index], int(len(z[index])/2)
                )

                tree, pred_features_tree = (self.decode_eval((h_parent.unsqueeze(0), c_parent.unsqueeze(
                    0)), None, idx_to_label, placeholderid_to_nameid))

                trees.append(tree)
                pred_features.append(pred_features_tree)


            return trees, pred_features

    def decode_train(self, iteration, z, h_p, c_p, h_s, c_s, target, individual_losses, accuracies):
        batch_size = z.shape[0]
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
                                       node_order, parent_indices, vocabs, z)

            h_pred = torch.tanh(self.U_parent(h_parent) +
                                self.U_sibling(h_prev_sibling))

            # Calculate parent and sibling loss
            parent_loss = self.bce_loss(
                self.depth_pred(h_pred), is_parent) / total_nodes
            sibling_loss = self.bce_loss(
                self.width_pred(h_pred), has_sibling) / total_nodes

            loss += parent_loss + sibling_loss
            individual_losses['PARENT'] += parent_loss.item()
            individual_losses['SIBLING'] += sibling_loss.item()
            accuracies['PARENT'] += sum(
                (self.sigmoid(self.depth_pred(h_pred)) >= 0.5) == (is_parent == 1))
            accuracies['SIBLING'] += sum(
                (self.sigmoid(self.width_pred(h_pred)) >= 0.5) == (has_sibling == 1))

            # Get true label values
            label = features[current_nodes_indices].long()
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

                        accuracies[k] += sum(torch.argmax(label_pred, dim=-1)
                                             == label[vocabs_mask == k].view(-1))

                    else:
                        label_pred = prediction_layer(h_pred[vocabs_mask == k])

                        # Calculate cross entropy loss of label prediction
                        label_loss = self.label_losses[k]((label_pred
                                                           + self.offset_parent(is_parent[vocabs_mask == k])
                                                           + self.offset_sibling(has_sibling[vocabs_mask == k])), label[vocabs_mask == k].view(-1)) / sum(target['vocabs'] == k)

                        accuracies[k] += sum(torch.argmax(self.softmax(label_pred
                                                                       + self.offset_parent(is_parent[vocabs_mask == k])
                                                                       + self.offset_sibling(has_sibling[vocabs_mask == k])), dim=-1) == label[vocabs_mask == k].view(-1))

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
                        # Compute hidden and cell values of current nodes
                        h_parent, c_parent = self.lstm_parent(
                            emb_label, (h_parent[vocabs_mask == k], c_parent[vocabs_mask == k]))
                        h_sibling, c_sibling = self.lstm_sibling(
                            emb_label, (h_prev_sibling[vocabs_mask == k], c_prev_sibling[vocabs_mask == k]))

                        # Update the hidden and cell values matrices
                        h_p[current_nodes_indices[vocabs_mask == k]] = h_parent
                        c_p[current_nodes_indices[vocabs_mask == k]] = c_parent

                        h_s[current_nodes_indices[vocabs_mask == k]] = h_sibling
                        c_s[current_nodes_indices[vocabs_mask == k]] = c_sibling

                    else:
                        # Compute hidden and cell values of current nodes for previous siblings only (since we are not parents in the leafs)
                        if self.params['INDIV_LAYERS_VOCABS']:
                            h, c = self.leaf_lstms_sibling[k](
                                emb_label,
                                (h_prev_sibling[vocabs_mask == k],
                                 c_prev_sibling[vocabs_mask == k]))
                        else:
                            h, c = self.lstm_sibling(
                                emb_label, (h_prev_sibling[vocabs_mask == k], c_prev_sibling[vocabs_mask == k]))

                        # Update hidden and cell values matrices for siblings only (leafs cannot be parents)
                        h_s[current_nodes_indices[vocabs_mask == k]] = h
                        c_s[current_nodes_indices[vocabs_mask == k]] = c

        return loss

    def get_hidden_values(self, iteration, adj_list, edge_order, h_p, c_p, h_s, c_s, sibling_index,
                          first_sibling_indices, current_indices, node_order, parent_indices, vocabs, z):
        # At sibling index 0, there should not be any previous siblings
        if sibling_index == 0:
            num_first_siblings = len(first_sibling_indices)
            # Initialize to hidden, cell of siblings to zero
            h_prev_sibling = torch.zeros(
                num_first_siblings, self.latent_dim, device=self.device)
            c_prev_sibling = torch.zeros(
                num_first_siblings, self.latent_dim, device=self.device)

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
            current_nodes_indices = current_indices[[
                ind + 1 for ind in indices]]

        # Iteration 0: Root node, so there are no parents
        if iteration == 0:
            h_parent, c_parent = torch.split(z, int(z.shape[-1]/2), dim=-1)
        else:
            h_parent = h_p[parent_indices_siblings, :]
            c_parent = c_p[parent_indices_siblings, :]

        adj_list_curr = adj_list[edge_order == iteration, :]
        sib = list(first_sibling_indices) + [len(parent_indices)]
        vocabs_mask = np.atleast_1d(vocabs[current_nodes_indices.cpu()])

        is_parent = torch.tensor([[1.] if index in adj_list_curr[:, 0] else [
                                 0.] for index in current_nodes_indices], device=self.device)
        has_sibling = torch.tensor([[1.] if j-i - 1 > sibling_index else [0.]
                                    for i,  j in zip(sib[:-1], sib[1:]) if j-i > sibling_index], device=self.device)

        return h_parent, c_parent, h_prev_sibling, c_prev_sibling, is_parent, has_sibling, current_nodes_indices, vocabs_mask

    def decode_eval(self, parent_state, sibling_state, idx_to_label, placeholderid_to_nameid, features=None, parent_node=None, iteration=0):
        h_parent, c_parent = parent_state

        if sibling_state is not None:
            h_prev_sibling, _ = sibling_state
        else:
            # Initialize to hidden, cell of siblings to zero
            h_prev_sibling = torch.zeros(
                1, self.latent_dim, device=self.device)

        h_pred = torch.tanh(self.U_parent(h_parent) +
                            self.U_sibling(h_prev_sibling))

        # Probability of the node having children
        p_parent = self.sigmoid(self.depth_pred(h_pred))
        # Probability of the node having successor children
        p_sibling = self.sigmoid(self.width_pred(h_pred))

        # Sample is_parent and has_sibling from predicted probability of parent/sibling
        is_parent = torch.distributions.bernoulli.Bernoulli(p_parent).sample()
        has_sibling = torch.distributions.bernoulli.Bernoulli(
            p_sibling).sample()

        # TODO Investigate why this is needed because it should not be needed
        if parent_node is None:
            is_parent = torch.tensor([1.], device=self.device)

        # Could also simply use > 0.5 instead OR TODO BEAM SEARCH
        # is_parent = torch.tensor(1) if p_parent > 0.5 else torch.tensor(0)
        # has_sibling = torch.tensor(1) if p_sibling > 0.5 else torch.tensor(0)
        if is_parent:
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
                label_pred + self.offset_parent(is_parent) + self.offset_sibling(has_sibling))

        # TODO sample the predicted label instead of argmax
        predicted_label = torch.distributions.categorical.Categorical(torch.exp(predicted_label)).sample()
        # topk_log_prob, topk_indexes = predicted_label.topk(3, sorted=True)

        # predicted_label = torch.argmax(predicted_label, dim=-1)

        # Build tree: Add node to tree
        if parent_node is None:
            node = Node(predicted_label.item(), is_reserved=True, parent=None)
        elif node_type == 'NAME':
            if predicted_label in placeholderid_to_nameid:
                node = Node(placeholderid_to_nameid[predicted_label.item(
                )], is_reserved=False, parent=parent_node)
            else:
                # TODO Fix this as this is not correct!
                node = Node(predicted_label.item(
                ), is_reserved=False, parent=parent_node)
        else:
            node = Node(predicted_label.item(
            ), is_reserved=True if is_parent else False, parent=parent_node)

        if features is None:
            features = [predicted_label.item()]
        else:
            features.append(predicted_label.item())

        if is_parent or not self.params['INDIV_LAYERS_VOCABS']:
            emb_label = self.embedding_layers[node_type](
                predicted_label).view(-1, self.params['EMBEDDING_DIM'])
        else:
            emb_label = self.embedding_layers[node_type](
                predicted_label).view(-1, self.params['LEAF_EMBEDDING_DIM'])

        # If we predict a next sibling
        if has_sibling:
            if is_parent or not self.params['INDIV_LAYERS_VOCABS']:
                # Calculate next hidden sibling state
                sibling_state = self.lstm_sibling(emb_label, sibling_state)
            else:
                # Calculate next hidden sibling state
                sibling_state = self.leaf_lstms_sibling[node_type](
                    emb_label, sibling_state)

            # print(parent_state[0][0][0].item(), sibling_state[0][0][0].item(), torch.argmax(predicted_label, dim=-1).item(), node_type)
            # Pass the same parent state, but updated sibling state
            self.decode_eval(parent_state, sibling_state, idx_to_label,
                             placeholderid_to_nameid, features, parent_node, iteration + 1)

        # We set the created node as the parent node
        parent_node = node

        # If we predict we are a parent, continue with children
        if is_parent:
            # update parent state and parent_node of tree
            parent_state = self.lstm_parent(emb_label, parent_state)

            # Pass new parent state and no sibling state as we start with the first sibling
            self.decode_eval(parent_state, None, idx_to_label,
                             placeholderid_to_nameid, features, parent_node, iteration + 1)

        # If we are done, return the root node (which contains the entire tree)
        return parent_node, features



class BeamSearch():
    def __init__(self):
        super().__init__()
        candidates = []
