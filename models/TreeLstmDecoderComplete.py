import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.TreeNode import Node
from utils.Sampling import Sampling
from time import time
from model_utils.modules import AddGate, PredictiveHidden, TreeTopologyPred, MultiLayerLSTMCell
from model_utils.state import TreeState
from config.vae_config import ex


# TODO make flag to combine output of LSTM cell and hidden state for predictive purposes or just take the hidden OR cell state


class TreeLstmDecoderComplete(nn.Module):
    @ex.capture
    def __init__(self,
                 device, 
                 embedding_layers,
                 vocabulary,
                 loss_weights,
                 embedding_dim, 
                 rnn_hidden_size, 
                 latent_dim, 
                 use_cell_output_lstm, 
                 num_rnn_layers_dec,
                 indiv_embed_layers,
                 max_name_tokens
                 ):
        super().__init__()

        self.device = device
        self.vocabulary = vocabulary
        self.loss_weights = loss_weights
        self.rnn_hidden_size = rnn_hidden_size
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.use_cell_output_lstm = use_cell_output_lstm
        self.num_rnn_layers_dec = num_rnn_layers_dec
        self.indiv_embed_layers = indiv_embed_layers
        self.max_name_tokens = max_name_tokens

        # Shared embedding layers (shared with encoder)
        self.embedding_layers = embedding_layers

        # Latent to hidden layer -> transform z from latent dim to hidden size
        self.latent2hidden = nn.Linear(self.latent_dim, rnn_hidden_size)

        # Doubly recurrent network layers: parent and sibling RNNs
        self.rnns_parent = MultiLayerLSTMCell(embedding_dim, rnn_hidden_size, num_rnn_layers_dec)
        self.rnns_sibling = MultiLayerLSTMCell(embedding_dim, rnn_hidden_size, num_rnn_layers_dec)

        self.pred_hidden_state = PredictiveHidden(rnn_hidden_size)

        self.add_gate = AddGate(rnn_hidden_size)

        self.tree_topology_pred = TreeTopologyPred(rnn_hidden_size)

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
        # Initialize prediction layers  
        for vocab_name in self.vocabulary.token_counts.keys():
            if vocab_name == 'LITERAL':
                # Adaptive softmax loss does not need prediction layer, much faster approach
                    # for calculating softmax with highly imbalanced vocabs
                    # cutoffs: 10: 71.1%, 11-100: 18.8%, 101-1000: 7.8%, rest: 2.2%
                    self.label_losses[vocab_name] = nn.AdaptiveLogSoftmaxWithLoss(
                                                        self.rnn_hidden_size,
                                                        self.vocabulary.get_vocab_size(vocab_name),
                                                        cutoffs=[10, 100, 1000],
                                                        div_value=3.0)
            else:
                # Prediction layer
                self.prediction_layers[vocab_name] = nn.Linear(self.rnn_hidden_size, self.vocabulary.get_vocab_size(vocab_name))
                # cross entropy loss
                self.label_losses[vocab_name] = nn.CrossEntropyLoss(weight=self.loss_weights[vocab_name], reduction='sum')


    def forward(self, z, target=None, oov_name_token2index=None):
        """
        @param z: (batch_size, LATENT_DIM) -> latent vector(s)

        @param target: dictionary containing tree information -> node_order_topdown, edge_order_topdown,
                                                  features, adjacency_list and vocabs

        @param idx_to_label: dictionary containing mapping from ids to labels, needed for evaluation
        """

        # We are training and we can do teacher forcing and batch processing
        if target is not None:
            # Keep track of the loss
            total_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            individual_losses = {}
            accuracies = {}
            loss_types = list(self.label_losses.keys()) + \
                ['PARENT', 'SIBLING', 'IS_RES']

            for loss_type in loss_types:
                individual_losses[loss_type] = 0
                accuracies[loss_type] = 0

            node_order = target['node_order_topdown']

            # Total number of nodes of all trees in the batch
            total_nodes = node_order.shape[0]

            # Hidden and cell states of RNNs for parent and sibling
            h_p = []
            c_p = []

            h_s = []
            c_s = []

            for _ in range(self.num_rnn_layers_dec):
                # h and c states for every node in the batch for parent lstm
                h_p.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))
                c_p.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))

                # h and c states for every node in the batch for sibling lstm
                h_s.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))
                c_s.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))

            # Iterate over the levels of the tree -> top down
            for iteration in range(node_order.max() + 1):
                loss = self.decode_train(iteration, z, target, h_p, c_p, h_s, c_s, individual_losses, accuracies)
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
        elif oov_name_token2index is not None:
            trees = self.start_decode_eval(z, oov_name_token2index)
            return trees


    def decode_train(self, iteration, z, data, h_p, c_p, h_s, c_s, individual_losses, accuracies):
        loss = 0

        node_order = data['node_order_topdown']
        edge_order = data['edge_order_topdown']
        edge_order_sib = data['edge_order_topdown_sib']
        features = data['features']
        features_combined = data['features_combined']
        adjacency_list = data['adjacency_list'].cpu()
        adjacency_list_sib = data['adjacency_list_sib'].cpu()
        vocabs = np.array(data['vocabs'])

        if iteration == 0:
            hidden = self.latent2hidden(z)

            if self.use_cell_output_lstm:
                h_parent, c_parent = torch.split(hidden, int(hidden.shape[-1]/2), dim=-1)
            else:
                h_parent = hidden
                c_parent = torch.zeros((z.shape[0], self.rnn_hidden_size), device=self.device)


             # TODO this should be what we are supposed to do, but what we do below seems to be working better
            # emb_label = self.embedding_layers['RES'](tree_state.features[tree_state.node_order == 0]).view(-1, self.params['EMBEDDING_DIM'])

            # hidden_states = [(h_parent, c_parent) for _ in range(self.params['NUM_LSTM_LAYERS'])]
            # new_parent_states = self.rnns_parent(emb_label, hidden_states)

            # for idx, state in enumerate(new_parent_states):    
            #     tree_state.h_p[idx][tree_state.node_order == 0] = state[0]
            #     tree_state.c_p[idx][tree_state.node_order == 0] = state[1]


            # FOR SOME REASON THIS SEEMS TO BE WORKING BETTER
            for i in range(self.num_rnn_layers_dec):
                # Update the hidden and cell values matrices
                h_p[i][node_order == 0] = h_parent
                c_p[i][node_order == 0] = c_parent


        else:
            # node_mask is a tensor of size N x 1
            node_mask = (node_order == iteration).cpu()
            # edge_mask is a tensor of size E x 1
            edge_mask = edge_order == iteration - 1

            edge_mask_sib = edge_order_sib == iteration - 1

            parent_indices = adjacency_list[edge_mask, :][:, 0]

            curr_h_p = []
            curr_c_p = []

            for i in range(self.num_rnn_layers_dec):
                curr_h_p.append(h_p[i][parent_indices, :])
                curr_c_p.append(c_p[i][parent_indices, :])

            prev_sib_indices = adjacency_list_sib[edge_mask_sib, :][:, 0]

            curr_h_s = []
            curr_c_s = []

            for i in range(self.num_rnn_layers_dec):
                curr_h_s.append(h_s[i][prev_sib_indices, :])
                curr_c_s.append(c_s[i][prev_sib_indices, :])


            # Combine hidden and cell states from last RNN layer
            if self.use_cell_output_lstm:
                parent_state = torch.cat([curr_h_p[-1], curr_c_p[-1]], dim=-1)
                sibling_state = torch.cat([curr_h_s[-1], curr_c_s[-1]], dim=-1)
            else:
                parent_state = curr_h_p[-1]
                sibling_state = curr_h_s[-1]

            h_pred = self.pred_hidden_state(parent_state, sibling_state)

            parent_state_updated = parent_state + self.add_gate(sibling_state)

            if self.use_cell_output_lstm:
                curr_h_p[-1], curr_c_p[-1] = torch.split(parent_state_updated, int(parent_state_updated.shape[-1]/2), dim=-1)
            else:
                curr_h_p[-1] = parent_state_updated

            depth_pred, width_pred, res_pred = self.tree_topology_pred(h_pred)

            is_parent = torch.tensor(np.isin(np.argwhere(node_mask == True), adjacency_list[:, 0]).flatten(), dtype=torch.float16, device=self.device).unsqueeze(-1)
            has_sibling = torch.tensor(np.isin(np.argwhere(node_mask == True), adjacency_list_sib[:, 0]).flatten(), dtype=torch.float16, device=self.device).unsqueeze(-1)
            is_res = torch.tensor(vocabs[node_mask] == 'RES', dtype=torch.float16, device=self.device).unsqueeze(-1)

            # Calculate parent and sibling loss
            parent_loss = self.bce_loss(depth_pred, is_parent)
            sibling_loss = self.bce_loss(width_pred, has_sibling)

            # calculate res prediction loss
            res_loss = self.bce_loss(res_pred, is_res)


            loss += parent_loss + sibling_loss + res_loss

            individual_losses['PARENT'] += parent_loss.item()
            individual_losses['SIBLING'] += sibling_loss.item()
            individual_losses['IS_RES'] += res_loss.item()

            accuracies['PARENT'] += sum(
                (self.sigmoid(depth_pred) >= 0.5) == (is_parent == 1)).item()
            accuracies['SIBLING'] += sum(
                (self.sigmoid(width_pred) >= 0.5) == (has_sibling == 1)).item()

            accuracies['IS_RES'] += sum(
                (self.sigmoid(res_pred) >= 0.5) == (is_res == 1)).item()


            label = features[node_mask].long()

            # Iterate over possible node types and predict labels for each node type
            for k, prediction_layer in list(self.prediction_layers.items()) + [('LITERAL', '_')]:
                # Only do actual calculations when the amount of nodes for this node type > 0
                if len(h_pred[vocabs[node_mask] == k]) > 0:
                    # Get label predictions
                    if k == 'LITERAL':
                        # TODO look into how we can incorporate offset parent and offset sibling here (see below)
                        # or if it even influences our loss since we work with the vocab types so maybe we can remove it altogether
                        label_pred, label_loss = self.label_losses[k](
                            h_pred[vocabs[node_mask] == k], label[vocabs[node_mask] == k].view(-1))

                        accuracies[k] += sum(self.label_losses[k].predict(h_pred[vocabs[node_mask] == k])
                                            == label[vocabs[node_mask] == k].view(-1)).item()

                    else:
                        label_pred = prediction_layer(h_pred[vocabs[node_mask] == k])

                        # Calculate cross entropy loss of label prediction
                        label_loss = self.label_losses[k]((label_pred
                                                        # + self.offset_parent(is_parent[vocabs_mask == k])
                                                        # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                        ), label[vocabs[node_mask] == k].view(-1)) # / sum(tree_state.vocabs == k)


                        accuracies[k] += sum(torch.argmax(self.softmax(label_pred
                                                                    # + self.offset_parent(is_parent[vocabs_mask == k])
                                                                    # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                                    ), dim=-1) == label[vocabs[node_mask] == k].view(-1)).item()
                        
                        # print(k, h_pred[vocabs[node_mask] == k], parent_state, sibling_state)
                        # pred_labels = torch.argmax(self.softmax(label_pred), dim=-1)
                        # print(iteration, k, pred_labels[pred_labels != label[vocabs[node_mask] == k].view(-1)].tolist(), pred_labels.tolist(), label[vocabs[node_mask] == k].view(-1).tolist()) #, self.softmax(label_pred).topk(3, sorted=True)[1], self.softmax(label_pred).topk(3, sorted=True)[0], h_pred[:, :5],)
                    loss += label_loss #* max(-100 * iteration + 200, 1)

                    individual_losses[k] += label_loss.item()

                    if self.indiv_embed_layers:
                        self.update_rnn_state(h_p, c_p, h_s, c_s, features, k, curr_h_p, curr_c_p, curr_h_s, curr_c_s, vocabs, node_mask)

            if not self.indiv_embed_layers:
                self.update_rnn_state(h_p, c_p, h_s, c_s, features_combined, 'ALL', curr_h_p, curr_c_p, curr_h_s, curr_c_s, vocabs, node_mask)


        return loss


    def update_rnn_state(self, h_p, c_p, h_s, c_s, features, node_type, curr_h_p, curr_c_p, curr_h_s, curr_c_s, vocabs, node_mask):
        # Get true label values
        label = features[node_mask].long()


        # Calculate embedding of true label -> teacher forcing

        if self.indiv_embed_layers:
            mask = vocabs[node_mask] == node_type
        else:
            mask = [True] * len(vocabs[node_mask])       


        emb_label = self.embedding_layers[node_type](label[mask]).view(-1, self.embedding_dim)
        

        parent_states = []
        sibling_states = []

        # Get the hidden and cell values of parents and previous siblings
        for i in range(self.num_rnn_layers_dec):
            parent_state_hidden = curr_h_p[i][mask]
            parent_state_cell = curr_c_p[i][mask]
            parent_states.append((parent_state_hidden, parent_state_cell))

            sibling_state_hidden = curr_h_s[i][mask]
            sibling_state_cell = curr_c_s[i][mask]
            sibling_states.append((sibling_state_hidden, sibling_state_cell))

        if 'RES' in node_type or node_type == 'ALL':
            # Compute hidden and cell values of current nodes
            new_parent_states = self.rnns_parent(emb_label, parent_states)
            new_sibling_states = self.rnns_sibling(emb_label, sibling_states)

        else:
            # Compute hidden and cell values of current nodes for previous siblings only (since we are not parents in the leafs)
            new_parent_states = [None for _ in range(self.num_rnn_layers_dec)]
            new_sibling_states = self.rnns_sibling(emb_label, sibling_states)

        
        # Update the hidden and cell values matrices
        state_mask = np.argwhere(node_mask == True).flatten()[mask]
        for idx, (p_state, s_state) in enumerate(zip(new_parent_states, new_sibling_states)):
            if 'RES' in node_type or node_type == 'ALL':
                h_p[idx][state_mask] = p_state[0]
                c_p[idx][state_mask] = p_state[1]

            h_s[idx][state_mask] = s_state[0]
            c_s[idx][state_mask] = s_state[1]


    def start_decode_eval(self, z, oov_name_token2index):
        oov_name_index2token = []

        hidden = self.latent2hidden(z)            

        for index in range(hidden.shape[0]):
            if oov_name_token2index is not None:
                oov_name_index2token.append({
                    v: k for k, v in oov_name_token2index[index].items()
                })
            else:
                oov_name_index2token = None

        rnn_empty_init = torch.zeros((z.shape[0], self.rnn_hidden_size), device=self.device)


        if self.use_cell_output_lstm:
            h_parent, c_parent = torch.split(hidden, int(hidden.shape[-1]/2), dim=-1)
        else:
            h_parent = hidden
            c_parent = rnn_empty_init

        parent_state = [(h_parent, c_parent) for _ in range(self.num_rnn_layers_dec)]
        sibling_state = [(rnn_empty_init, rnn_empty_init) for _ in range(self.num_rnn_layers_dec)]

        trees = self.decode_eval(parent_state, sibling_state, oov_name_index2token)

        return trees


    def decode_eval(self, parent_state, sibling_state, oov_name_index2token, parent_nodes=None, iteration=0):
        if iteration > 50:
            print(f'Stopped due to recursion level going over {iteration} iterations')
            return parent_nodes


        batch_size = parent_state[0][0].shape[0]
        # If we are at the root
        if parent_nodes is None:

            # Get the root node ID
            root_id = self.vocabulary.token2index['RES']['root']

            # Create root node of the tree
            parent_nodes = [Node(root_id, is_reserved=True, parent=None) for _ in range(batch_size)]

            # Update the parent state
            # emb_label = self.embedding_layers['RES'](torch.tensor([root_id], device=self.device)).view(-1, self.params['EMBEDDING_DIM'])

            # parent_state = self.rnns_parent(emb_label, parent_state)

            # for i in range(self.params['NUM_LSTM_LAYERS']):
            #     if i == 0:
            #         parent_state[i] = self.lstms_parent[i](emb_label, parent_state[i])
            #     else:
            #         parent_state[i] = self.lstms_parent[i](parent_state[i-1][0], None)


            # Pass new parent state and no sibling state as we start with the first sibling
            self.decode_eval(parent_state, sibling_state, oov_name_index2token, parent_nodes)

        else:
            h_parent, c_parent = parent_state[-1]

            h_prev_sibling, c_prev_sibling = sibling_state[-1]

            if self.use_cell_output_lstm:
                h_parent = torch.cat([h_parent, c_parent], dim=-1)
                h_prev_sibling = torch.cat([h_prev_sibling, c_prev_sibling], dim=-1)

            h_pred = self.pred_hidden_state(h_parent, h_prev_sibling)


            # sibling add gate to add to parent state           
            parent_state_updated = h_parent + self.add_gate(h_prev_sibling)


            if self.use_cell_output_lstm:
                parent_state_updated = torch.split(parent_state_updated, int(parent_state_updated.shape[-1]/2), dim=-1)


            depth_pred, width_pred, res_pred = self.tree_topology_pred(h_pred)

            # Probability of the node having children
            p_parent = self.sigmoid(depth_pred)
            # Probability of the node having successor children
            p_sibling = self.sigmoid(width_pred)

            # Probability of the node being a reserved c++ token
            p_res = self.sigmoid(res_pred)

            # Sample is_parent and has_sibling from predicted probability of parent/sibling
            # is_parent = (p_parent >= 0.5).view(-1)
            # has_sibling = (p_sibling >= 0.5).view(-1)
            is_parent = (torch.distributions.bernoulli.Bernoulli(p_parent).sample() == 1).view(-1)
            has_sibling = (torch.distributions.bernoulli.Bernoulli(p_sibling).sample() == 1).view(-1)

            is_res = (p_res >= 0.5).view(-1)

            """
                Node types:
                    - NAME:     0
                    - RES:      1
                    - LITERAL:  2
                    - TYPE:     3
            """

            node_types = torch.zeros(batch_size, device=self.device)

            parent_labels = []

            for parent_node in parent_nodes:
                if parent_node.token in self.vocabulary.index2token['RES']:
                    parent_labels.append(self.vocabulary.index2token['RES'][parent_node.token])
                else:
                    parent_labels.append('default token')

            # parent_labels = np.array([self.vocabulary.index2token['RES'][parent_node.token] for parent_node in parent_nodes])

            literal_indices = [idx for idx, label in enumerate(parent_labels) if 'LITERAL' in label]
            type_indices = [idx for idx, label in enumerate(parent_labels) if 'TYPE' == label]

            node_types[is_res] = 1
            node_types[literal_indices] = 2
            node_types[type_indices] = 3

            name_indices = node_types == 0

            label_logits_res = self.prediction_layers['RES'](h_pred[is_res])
            label_logits_type = self.prediction_layers['TYPE'](h_pred[type_indices])
            label_logits_name = self.prediction_layers['NAME'](h_pred[name_indices])
            label_logits_literal = self.label_losses['LITERAL'].log_prob(h_pred[literal_indices])

            predicted_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            predicted_labels[is_res] = Sampling.sample(label_logits_res, temperature=0.9, top_k=40, top_p=0.9).view(-1)
            predicted_labels[type_indices] = Sampling.sample(label_logits_type, temperature=0.9, top_k=40, top_p=0.9).view(-1)
            predicted_labels[name_indices] = Sampling.sample(label_logits_name, temperature=0.9, top_k=40, top_p=0.9).view(-1)
            predicted_labels[literal_indices] = Sampling.sample(label_logits_literal, temperature=0.9, top_k=40, top_p=0.5).view(-1)

            nodes = []

            for idx, predicted_label in enumerate(predicted_labels):
                # Build tree: Add node to tree
                if node_types[idx] == 0:
                    if predicted_label.item() < self.max_name_tokens:
                        node = Node(predicted_label.item(), is_reserved=False, parent=parent_nodes[idx])
                    elif oov_name_index2token is not None and predicted_label.item() in oov_name_index2token[idx]:
                        node = Node(oov_name_index2token[idx][predicted_label.item(
                        )], is_reserved=False, parent=parent_nodes[idx])
                    else:
                        node = Node(f'NAME_{predicted_label.item()}', is_reserved=False, parent=parent_nodes[idx])
                else:
                    node = Node(predicted_label.item(
                    ), is_reserved=True if node_types[idx] == 1 else False, parent=parent_nodes[idx])

                nodes.append(node)


            emb_labels = torch.zeros(batch_size, self.embedding_dim, device=self.device)


            for node_type, emb_layer in self.embedding_layers.items():
                if node_type == 'RES':
                    mask = is_res
                elif node_type == 'LITERAL':
                    mask = literal_indices
                elif node_type == 'TYPE':
                    mask = type_indices
                else:
                    mask = name_indices

                emb_labels[mask] = self.embedding_layers[node_type](
                        predicted_labels[mask]).view(-1, self.embedding_dim)


            total_sibling_state = []
            total_parent_state = []

            empty_init = torch.zeros(batch_size, self.rnn_hidden_size, device=self.device)

            if torch.count_nonzero(has_sibling) > 0:
                new_sibling_state = self.rnns_sibling(emb_labels, sibling_state)   
            else:
                new_sibling_state = [(empty_init, empty_init) for _ in range(self.num_rnn_layers_dec)]

            new_parent_state = parent_state[:]

            if self.use_cell_output_lstm: 
                new_parent_state[-1] = parent_state_updated
            else:
                new_parent_state[-1] = (parent_state_updated, new_parent_state[-1][1])


            if torch.count_nonzero(is_parent) > 0:
                new_parent_state = self.rnns_parent(emb_labels, new_parent_state)
            else:
                new_parent_state = [(empty_init, empty_init) for _ in range(self.num_rnn_layers_dec)]


            empty_init = torch.zeros(torch.count_nonzero(is_parent), self.rnn_hidden_size, device=self.device)


            for i in range(self.num_rnn_layers_dec):
                p_state_h_par = new_parent_state[i][0][is_parent]
                p_state_c_par = new_parent_state[i][1][is_parent]

                p_state_h_sib = parent_state[i][0][has_sibling]
                p_state_c_sib = parent_state[i][1][has_sibling]

                p_state_h = torch.cat([p_state_h_par, p_state_h_sib])
                p_state_c = torch.cat([p_state_c_par, p_state_c_sib])

                total_parent_state.append((p_state_h, p_state_c))


                s_state_h_par = empty_init
                s_state_c_par = empty_init

                s_state_h_sib = new_sibling_state[i][0][has_sibling]
                s_state_c_sib = new_sibling_state[i][1][has_sibling]

                p_state_h = torch.cat([s_state_h_par, s_state_h_sib])
                p_state_c = torch.cat([s_state_c_par, s_state_c_sib])


                total_sibling_state.append((p_state_h, p_state_c))


            parent_nodes = [node for is_p, node in zip(is_parent, nodes) if is_p] + [node for has_s, node in zip(has_sibling, parent_nodes) if has_s]
            if oov_name_index2token is not None: 
                oov_name_index2token = [vocab for is_p, vocab in zip(is_parent, oov_name_index2token) if is_p] +  [vocab for has_s, vocab in zip(has_sibling, oov_name_index2token) if has_s]

            if torch.count_nonzero(has_sibling) > 0 or torch.count_nonzero(is_parent) > 0:
                self.decode_eval(total_parent_state, total_sibling_state, oov_name_index2token, parent_nodes, iteration + 1)


        # If we are done, return the parent nodes (which contain the entire trees)
        return parent_nodes
