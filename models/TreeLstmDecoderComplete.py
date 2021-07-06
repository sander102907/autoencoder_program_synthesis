from os import name
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.TreeNode import Node
from utils.Sampling import Sampling
from model_utils.modules import AddGate, PredictiveHidden, TreeTopologyPred, MultiLayerLSTMCell
from model_utils.distance_functions import CosineDistance
from model_utils.adaptive_softmax_pytorch import AdaptiveLogSoftmaxWithLoss
from config.vae_config import ex
from random import choice

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
                 dropout,
                 recurrent_dropout,
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
        self.rnns_parent = MultiLayerLSTMCell(embedding_dim, rnn_hidden_size, num_rnn_layers_dec, recurrent_dropout)
        self.rnns_sibling = MultiLayerLSTMCell(embedding_dim, rnn_hidden_size, num_rnn_layers_dec, recurrent_dropout)

        self.pred_hidden_state = PredictiveHidden(rnn_hidden_size)

        self.add_gate = AddGate(rnn_hidden_size)

        self.tree_topology_pred = TreeTopologyPred(rnn_hidden_size)

        # Learnable weights to incorporate topology information in label prediction
        self.offset_parent = nn.Linear(1, 1, bias=True)
        self.offset_sibling = nn.Linear(1, 1, bias=True)


        self.name_weights = nn.Linear(rnn_hidden_size, self.embedding_dim, bias=True)

        # Leaf lstms, if we want individual RNNs for leaf nodes
        self.leaf_lstms_sibling = nn.ModuleDict({})

        # Prediction layers for each vocab
        self.prediction_layers = nn.ModuleDict({})

        # Binary cross entropy loss for computing loss of topology
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # Loss functions for labels for each vocab so we can specify weights if needed
        self.label_losses = nn.ModuleDict({})

        self.dropout = nn.Dropout(dropout)


        # For evaluation/testing/generating:

        # sigmoid to predict topology of the tree (width and depth)
        self.sigmoid = nn.Sigmoid()

        # softmax to get probability distribution for labels of nodes
        self.softmax = nn.LogSoftmax(dim=-1)

        # Cosine similarity to get the similarity between name declarations and references
        self.dist_function = CosineDistance(dim=-1)

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function, reduction='sum', margin=1)

        # Initialize layers
        self.init_layers()


    def init_layers(self):
        # Initialize prediction layers  
        for vocab_name in self.vocabulary.token_counts.keys():
            if vocab_name == 'LITERAL':
                # Adaptive softmax loss does not need prediction layer, much faster approach
                    # for calculating softmax with highly imbalanced vocabs
                    # cutoffs: 10: 71.1%, 11-100: 18.8%, 101-1000: 7.8%, rest: 2.2%
                    self.label_losses[vocab_name] = AdaptiveLogSoftmaxWithLoss(
                                                        self.rnn_hidden_size,
                                                        self.vocabulary.get_vocab_size(vocab_name),
                                                        cutoffs=[10, 100, 300],
                                                        div_value=3.0)

            elif vocab_name == 'NAME':
                self.label_losses[vocab_name] = None
            else:
                # Prediction layer
                self.prediction_layers[vocab_name] = nn.Linear(self.rnn_hidden_size, self.vocabulary.get_vocab_size(vocab_name))
                # cross entropy loss
                self.label_losses[vocab_name] = nn.CrossEntropyLoss(weight=self.loss_weights[vocab_name], reduction='sum')


    def forward(self, z, inp=None, names_token2index=None, temperature=None, top_k=None, top_p=None):
        """
        @param z: (batch_size, LATENT_DIM) -> latent vector(s)

        @param target: dictionary containing tree information -> node_order_topdown, edge_order_topdown,
                                                  features, adjacency_list and vocabs

        @param idx_to_label: dictionary containing mapping from ids to labels, needed for evaluation
        """

        # We are training and we can do teacher forcing and batch processing
        if inp is not None:
            # Keep track of the loss
            total_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            individual_losses = {}
            accuracies = {}
            loss_types = list(self.label_losses.keys()) + \
                ['PARENT', 'SIBLING', 'IS_RES']

            for loss_type in loss_types:
                individual_losses[loss_type] = 0
                accuracies[loss_type] = 0

            node_order = inp['node_order_topdown']

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

            # Save predictive hidden states for name clustering
            # h_pred_states = torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device)

            declared_names = [{} for _ in range(len(inp['tree_sizes']))]

            # Iterate over the levels of the tree -> top down
            for iteration in range(node_order.max() + 1):
                loss = self.decode_train(iteration, z, inp, h_p, c_p, h_s, c_s, declared_names, individual_losses, accuracies)
                total_loss += loss

            for loss_type in loss_types:
                if loss_type in ['PARENT', 'SIBLING', 'IS_RES']:
                    accuracies[loss_type] = accuracies[loss_type] / (total_nodes - z.shape[0])
                    # print(loss_type, accuracies[loss_type])
                else:
                    if loss_type == 'RES':
                        # Correct for root node -> is not predicted
                        accuracies[loss_type] = accuracies[loss_type] / (sum(inp['vocabs'] == loss_type) - z.shape[0])
                    elif loss_type == 'NAME':
                        accuracies[loss_type] = accuracies[loss_type] / (sum(inp['vocabs'] == loss_type) - sum([len(tree) for tree in declared_names]))
                    else:
                        accuracies[loss_type] = accuracies[loss_type] / sum(inp['vocabs'] == loss_type)

            return total_loss, individual_losses, accuracies

         # We are evaluating and we cannot use training forcing and we generate tree by tree
        elif names_token2index is not None:
            trees = self.start_decode_eval(z, names_token2index, temperature, top_k, top_p)
            return trees


    def decode_train(self, iteration, z, data, h_p, c_p, h_s, c_s, declared_names, individual_losses, accuracies):
        loss = 0

        node_order = data['node_order_topdown']
        edge_order = data['edge_order_topdown']
        edge_order_sib = data['edge_order_topdown_sib']
        features = data['features']
        features_combined = data['features_combined']
        adjacency_list = data['adjacency_list'].cpu()
        adjacency_list_sib = data['adjacency_list_sib'].cpu()
        vocabs = np.array(data['vocabs'])
        tree_indices = data['tree_indices']

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

            prev_sib_indices = adjacency_list_sib[edge_mask_sib, :][:, 0]

            curr_h_p = []
            curr_c_p = []

            for i in range(self.num_rnn_layers_dec):
                curr_h_p.append(h_p[i][parent_indices, :])
                curr_c_p.append(c_p[i][parent_indices, :])


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

            h_pred = self.dropout(h_pred)

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

            accuracies['PARENT'] += torch.sum(
                (self.sigmoid(depth_pred) >= 0.5) == (is_parent == 1)).item()
            accuracies['SIBLING'] += torch.sum(
                (self.sigmoid(width_pred) >= 0.5) == (has_sibling == 1)).item()

            accuracies['IS_RES'] += torch.sum(
                (self.sigmoid(res_pred) >= 0.5) == (is_res == 1)).item()

            label = features[node_mask].long()


            # Iterate over possible node types and predict labels for each node type
            for k, prediction_layer in list(self.prediction_layers.items()) + [('LITERAL', '_'), ('NAME', '_')]:
                # Only do actual calculations when the amount of nodes for this node type > 0
                if len(h_pred[vocabs[node_mask] == k]) > 0:
                    # Get label predictions
                    if k == 'LITERAL':
                        # TODO look into how we can incorporate offset parent and offset sibling here (see below)
                        # or if it even influences our loss since we work with the vocab types so maybe we can remove it altogether
                        label_pred, label_loss = self.label_losses[k](
                            h_pred[vocabs[node_mask] == k], label[vocabs[node_mask] == k].view(-1))

                        accuracies[k] += torch.sum(self.label_losses[k].predict(h_pred[vocabs[node_mask] == k])
                                            == label[vocabs[node_mask] == k].view(-1)).item()


                        # Reduction is set to None for adaptive softmax with logit loss
                        label_loss = torch.sum(label_loss)

                        loss += label_loss
                        individual_losses[k] += label_loss.item()
                    elif k == 'NAME':
                        # Get the indices of the current name nodes we are evaluating
                        node_mask_indices = torch.where(torch.tensor(vocabs[node_mask] == k))[0]

                        # Loop over the name labels we are evaluating
                        for idx, l in zip(node_mask_indices, label[node_mask_indices]):
                            # Get the current node index (from all nodes, not just the selection)
                            current_node_idx = torch.where(node_mask)[0][idx]

                            # Get the declared names of the program we are evaluating
                            program_declared_names = declared_names[tree_indices[node_mask][idx]]

                            # Remove the declared names that are defined after the current node
                            program_declared_names_filt = {k: v for k, v in program_declared_names.items() if program_declared_names[k][0] < current_node_idx}

                            # If the name is already declared -> calculate loss: triplet between declaration as positive
                            #  and between other declarations as negative
                            if l.item() in program_declared_names_filt:
                                positive = self.name_weights(h_pred[idx]).unsqueeze(0)
                               
                                if len(program_declared_names_filt) > 1:
                                    # create a name prediction state to make the h_pred of the name similar to the h_pred of the declaration
                                    anchor = program_declared_names_filt[l.item()][1]
                                    # negative = torch.stack([v[1] for k, v in program_declared_names_filt.items() if k != l.item()])
                                    negative = choice([v[1] for k, v in program_declared_names_filt.items() if k != l.item()]).unsqueeze(0)

                                    name_ref_loss = self.triplet_loss(anchor, positive, negative)

                                    loss += name_ref_loss
                                    individual_losses[k] += name_ref_loss.item()

                                
                                distances = self.dist_function(positive, torch.stack([v[1] for v in program_declared_names_filt.values()]))
                                most_similar = list(program_declared_names_filt.keys())[torch.argmin(distances)]

                                # topk = min(3, len(program_declared_names))
                                # most_similar = torch.tensor(list(program_declared_names.keys()))[torch.topk(distances, k=topk, largest=False)[1]]
                                accuracies[k] +=  l.item() == most_similar
                            else:
                                program_declared_names[l.item()] = (current_node_idx, self.name_weights(h_pred[idx]))

                    else:
                        label_pred = prediction_layer(h_pred[vocabs[node_mask] == k])

                        # Calculate cross entropy loss of label prediction
                        label_loss = self.label_losses[k]((label_pred
                                                        # + self.offset_parent(is_parent[vocabs_mask == k])
                                                        # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                        ), label[vocabs[node_mask] == k].view(-1)) # / sum(tree_state.vocabs == k)


                        accuracies[k] += torch.sum(torch.argmax(self.softmax(label_pred
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

        if self.indiv_embed_layers:
            mask = vocabs[node_mask] == node_type
        else:
            mask = [True] * len(vocabs[node_mask])       


        # Calculate embedding of true label -> teacher forcing
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


    def start_decode_eval(self, z, names_token2index, temperature, top_k, top_p):
        names_index2token = []

        hidden = self.latent2hidden(z)            

        for index in range(hidden.shape[0]):
            if names_token2index is not None:
                names_index2token.append({
                    v: k for k, v in names_token2index[index].names.items()
                })
            else:
                names_index2token = None

        rnn_empty_init = torch.zeros((z.shape[0], self.rnn_hidden_size), device=self.device)


        if self.use_cell_output_lstm:
            h_parent, c_parent = torch.split(hidden, int(hidden.shape[-1]/2), dim=-1)
        else:
            h_parent = hidden
            c_parent = rnn_empty_init

        name_eval_nodes = NameEvalNodes(self.device)

        parent_state = [(h_parent, c_parent) for _ in range(self.num_rnn_layers_dec)]
        sibling_state = [(rnn_empty_init, rnn_empty_init) for _ in range(self.num_rnn_layers_dec)]

        trees = self.decode_eval(parent_state, sibling_state, names_index2token, name_eval_nodes, temperature, top_k, top_p)

        return trees


    def decode_eval(self,
                    parent_state, 
                    sibling_state, 
                    names_index2token, 
                    name_eval_nodes,
                    temperature,
                    top_k,
                    top_p, 
                    parent_nodes=None, 
                    declared_names=None, 
                    program_ids=None, 
                    sibling_path_offsets=None, 
                    iteration=0):

        if iteration > 50 and not name_eval_nodes.processing_names:
            print(f'Stopped due to recursion level going over {iteration} iterations')
            return parent_nodes


        batch_size = parent_state[0][0].shape[0]
        # If we are at the root
        if parent_nodes is None:
            # Get the root node ID
            root_id = self.vocabulary.token2index['RES']['root']

            # Create root node of the tree
            parent_nodes = [Node(root_id, is_reserved=True, parent=None) for _ in range(batch_size)]
            declared_names = [{} for _ in range(batch_size)]
            program_ids = [i for i in range(batch_size)]
            sibling_path_offsets = [[0] for _ in range(batch_size)]

            # Update the parent state
            # emb_label = self.embedding_layers['RES'](torch.tensor([root_id], device=self.device)).view(-1, self.params['EMBEDDING_DIM'])

            # parent_state = self.rnns_parent(emb_label, parent_state)

            # for i in range(self.params['NUM_LSTM_LAYERS']):
            #     if i == 0:
            #         parent_state[i] = self.lstms_parent[i](emb_label, parent_state[i])
            #     else:
            #         parent_state[i] = self.lstms_parent[i](parent_state[i-1][0], None)


            # Pass new parent state and no sibling state as we start with the first sibling
            self.decode_eval(parent_state, 
                            sibling_state, 
                            names_index2token, 
                            name_eval_nodes, 
                            temperature, top_k, 
                            top_p, parent_nodes, 
                            declared_names, 
                            program_ids, 
                            sibling_path_offsets)

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


            # Make sure we do not get infinitely long repeating siblings
            # So when we get > 8 consecutive siblings, slowly start reducing p_sib
            for p_sib, sib_path_offset in zip(p_sibling, sibling_path_offsets):
                if sib_path_offset[-1] > 8:
                    p_sib *= 1 - ((sib_path_offset[-1] - 8) / 10)


            # Sample is_parent and has_sibling from predicted probability of parent/sibling
            is_parent = (p_parent >= 0.5).view(-1)
            has_sibling = (p_sibling >= 0.5).view(-1)
            # is_parent = (torch.distributions.bernoulli.Bernoulli(p_parent).sample() == 1).view(-1)
            # has_sibling = (torch.distributions.bernoulli.Bernoulli(p_sibling).sample() == 1).view(-1)

            # print(p_sibling, sibling_path_offsets)

            is_res = (p_res >= 0.5).view(-1)


            n_types = {
                0: 'NAME',
                1: 'NAME_BUILTIN',
                2: 'RES',
                3: 'LITERAL',
                4: 'TYPE'
                }

            

            node_types = torch.zeros(batch_size, device=self.device)

            parent_labels = []

            for parent_node in parent_nodes:
                if parent_node.token in self.vocabulary.index2token['RES']:
                    parent_labels.append(self.vocabulary.index2token['RES'][parent_node.token])
                else:
                    parent_labels.append('default token')

            # print(parent_labels, '\n', p_parent, '\n',  p_sibling, '\n',  p_res, '\n\n')

            name_builtin_indices = [idx for idx, label in enumerate(parent_labels) if 'REF_BUILTIN' == label]
            literal_indices = [idx for idx, label in enumerate(parent_labels) if 'LITERAL' in label]
            type_indices = [idx for idx, label in enumerate(parent_labels) if 'TYPE' == label]

            node_types[name_builtin_indices] = 1
            node_types[is_res] = 2
            node_types[literal_indices] = 3
            node_types[type_indices] = 4

            name_indices = node_types == 0

            label_logits_res = self.prediction_layers['RES'](h_pred[is_res])
            label_logits_type = self.prediction_layers['TYPE'](h_pred[type_indices])
            label_logits_name_builtin = self.prediction_layers['NAME_BUILTIN'](h_pred[name_builtin_indices])
            label_logits_literal = self.label_losses['LITERAL'].log_prob(h_pred[literal_indices])


            predicted_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            predicted_labels[is_res] = Sampling.sample(label_logits_res, temperature, top_k, top_p).view(-1)
            predicted_labels[type_indices] = Sampling.sample(label_logits_type, temperature, top_k, top_p).view(-1)
            predicted_labels[name_builtin_indices] = Sampling.sample(label_logits_name_builtin, temperature, top_k, top_p).view(-1)
            predicted_labels[literal_indices] = Sampling.sample(label_logits_literal, temperature, top_k, top_p).view(-1)

            predicted_name_labels = []

            # print([(k, v[0]) for k,v in declared_names[0].items()], sibling_path_offsets)

            for idx, name_idx in enumerate(torch.where(name_indices)[0]):

                amt_declared_names = len([name_state for name_state in declared_names[program_ids[name_idx]].values() if self.is_declared(sibling_path_offsets[name_idx], name_state[0])])

                # If the parent is not a reference or the number of delcarations is 0
                if ('REF' not in parent_labels[name_idx] \
                    or amt_declared_names == 0) \
                    and len(list(declared_names[program_ids[name_idx]])) < self.vocabulary.get_vocab_size('NAME') - 1:

                    name_label = len(declared_names[program_ids[name_idx]])
                    declared_names[program_ids[name_idx]][name_label] = (sibling_path_offsets[name_idx], self.name_weights(h_pred[name_idx]))

                    predicted_name_labels.append(name_label)
                else:
                    names_in_program = torch.stack([name_state[1] for name_state in declared_names[program_ids[name_idx]].values() if self.is_declared(sibling_path_offsets[name_idx], name_state[0])])
                    current_name = self.name_weights(h_pred[name_idx])


                    distances = self.dist_function(current_name, names_in_program)
                    most_similar = list(declared_names[program_ids[name_idx]].keys())[torch.argmin(distances)]

                    predicted_name_labels.append(most_similar)

            predicted_labels[name_indices] = torch.tensor(predicted_name_labels, dtype=torch.long, device=self.device)

            nodes = []
            nodes_labels = []

            for idx, predicted_label in enumerate(predicted_labels):
                # Build tree: Add node to tree
                if node_types[idx] == 0:
                    if predicted_label.item() < self.max_name_tokens:
                        node = Node(predicted_label.item(), is_reserved=False, parent=parent_nodes[idx])
                    # elif oov_name_index2token is not None and predicted_label.item() in oov_name_index2token[idx]:
                    #     node = Node(oov_name_index2token[idx][predicted_label.item(
                    #     )], is_reserved=False, parent=parent_nodes[idx])
                    else:
                        node = Node(f'NAME_{predicted_label.item()}', is_reserved=False, parent=parent_nodes[idx])
                else:
                    node = Node(predicted_label.item(),
                                is_reserved=True if node_types[idx] == 2 else False,
                                parent=parent_nodes[idx])

                nodes.append(node)

                if node.token in self.vocabulary.index2token['RES']:
                    nodes_labels.append(self.vocabulary.index2token['RES'][node.token])
                else:
                    nodes_labels.append('default token')

            # If there are next children or siblings, recursively continue
            if torch.count_nonzero(has_sibling) > 0 or torch.count_nonzero(is_parent) > 0:

                if self.indiv_embed_layers:
                    emb_labels = torch.zeros(batch_size, self.embedding_dim, device=self.device)

                    for node_type, emb_layer in self.embedding_layers.items():
                        if node_type == 'RES':
                            mask = is_res
                        elif node_type == 'LITERAL':
                            mask = literal_indices
                        elif node_type == 'TYPE':
                            mask = type_indices
                        elif node_type == 'NAME_BUILTIN':
                            mask = name_builtin_indices
                        else:
                            mask = name_indices

                        emb_labels[mask] = emb_layer(predicted_labels[mask]).view(-1, self.embedding_dim)
                else:
                    # predicted_labels[is_res] = torch.tensor([self.vocabulary.token2index['ALL'][self.vocabulary.index2token['RES'][token.item()]] for token in predicted_labels[is_res]], dtype=torch.long, device=self.device)
                    # predicted_labels[type_indices] = torch.tensor([self.vocabulary.token2index['ALL'][self.vocabulary.index2token['TYPE'][token.item()]] for token in predicted_labels[type_indices]], dtype=torch.long, device=self.device)
                    # predicted_labels[name_builtin_indices] = torch.tensor([self.vocabulary.token2index['ALL'][self.vocabulary.index2token['NAME_BUILTIN'][token.item()]] for token in predicted_labels[name_builtin_indices]], dtype=torch.long, device=self.device)
                    # predicted_labels[literal_indices] = torch.tensor([self.vocabulary.token2index['ALL'][self.vocabulary.index2token['LITERAL'][token.item()]] for token in predicted_labels[literal_indices]], dtype=torch.long, device=self.device)
                    # predicted_labels[name_indices] = torch.tensor([self.vocabulary.token2index['ALL'][self.vocabulary.index2token['NAME'][token.item()]] for token in predicted_labels[name_indices]], dtype=torch.long, device=self.device)

                    try:
                        predicted_labels = torch.tensor(
                            [self.vocabulary.token2index['ALL'][self.vocabulary.index2token[n_types[n_type.item()]][token.item()]] for token, n_type in zip(predicted_labels, node_types)],
                            dtype=torch.long,
                            device=self.device
                        )
                    except Exception as e:
                        print(e, predicted_labels.tolist(), node_types.tolist(), declared_names[0].keys())
                        return

                    emb_labels = self.embedding_layers['ALL'](predicted_labels).view(-1, self.embedding_dim)

                # Lists to save the new parent and sibling states
                total_sibling_state = []
                total_parent_state = []


                total_sibling_state_names = []
                total_parent_state_names = []

                empty_init = torch.zeros(batch_size, self.rnn_hidden_size, device=self.device)

                # Get new siblings state, if there are any next siblings
                if torch.count_nonzero(has_sibling) > 0:
                    new_sibling_state = self.rnns_sibling(emb_labels, sibling_state)   
                else:
                    new_sibling_state = [(empty_init, empty_init) for _ in range(self.num_rnn_layers_dec)]


                # Create a copy of the parent state (we need the original for the next siblings)
                new_parent_state = parent_state[:]

                # Update the las layer of the parent state with the add gate update
                if self.use_cell_output_lstm: 
                    new_parent_state[-1] = parent_state_updated
                else:
                    new_parent_state[-1] = (parent_state_updated, new_parent_state[-1][1])


                # Get new parent state, if there are any next children
                if torch.count_nonzero(is_parent) > 0:
                    new_parent_state = self.rnns_parent(emb_labels, new_parent_state)
                else:
                    new_parent_state = [(empty_init, empty_init) for _ in range(self.num_rnn_layers_dec)]


                # Get current node types, to put name nodes in a queue for predicting last
                # node_types = torch.zeros(batch_size, device=self.device)

                # name_builtin_indices = [idx for idx, label in enumerate(nodes_labels) if 'REF_BUILTIN' == label]
                # literal_indices = [idx for idx, label in enumerate(nodes_labels) if 'LITERAL' in label]
                # type_indices = [idx for idx, label in enumerate(nodes_labels) if 'TYPE' == label]

                # node_types[name_builtin_indices] = 1
                # node_types[is_res] = 2
                # node_types[literal_indices] = 3
                # node_types[type_indices] = 4

                # name_indices = node_types == 0

                name_indices = torch.tensor([True if label in ['REF', 'NAME', 'IDENTIFIER', 'TYPE_REF'] else False for label in nodes_labels], device=self.device)

                empty_init = torch.zeros(torch.count_nonzero(is_parent & ~name_indices), self.rnn_hidden_size, device=self.device)
                empty_init_names = torch.zeros(torch.count_nonzero(is_parent & name_indices), self.rnn_hidden_size, device=self.device)


                for i in range(self.num_rnn_layers_dec):
                    # Get the parent states, for the next children nodes
                    p_state_h_par = new_parent_state[i][0][is_parent & ~name_indices]
                    p_state_c_par = new_parent_state[i][1][is_parent & ~name_indices]

                    # Get the parent states, for the next sibling nodes
                    # Note that we do not use the new parent state here
                    # As next siblings have the same parent
                    p_state_h_sib = parent_state[i][0][has_sibling]
                    p_state_c_sib = parent_state[i][1][has_sibling]

                    # Concatenate the parent states, children first then siblings
                    p_state_h = torch.cat([p_state_h_par, p_state_h_sib])
                    p_state_c = torch.cat([p_state_c_par, p_state_c_sib])

                    # Append parent states of this RNN layer to total state
                    total_parent_state.append((p_state_h, p_state_c))


                    p_state_h_names = new_parent_state[i][0][is_parent & name_indices]
                    p_state_c_names = new_parent_state[i][1][is_parent & name_indices]

                    total_parent_state_names.append((p_state_h_names, p_state_c_names))


                    # Get the sibling states, for the next children nodes
                    # Note, new children do not have any siblings
                    s_state_h_par = empty_init
                    s_state_c_par = empty_init

                    # Get the sibling states, for the next sibling nodes
                    s_state_h_sib = new_sibling_state[i][0][has_sibling]
                    s_state_c_sib = new_sibling_state[i][1][has_sibling]

                    # Concatenate the sibling states, children first, then siblings
                    s_state_h = torch.cat([s_state_h_par, s_state_h_sib])
                    s_state_c = torch.cat([s_state_c_par, s_state_c_sib])

                    # Append sibling states of this RNN layer to the total state
                    total_sibling_state.append((s_state_h, s_state_c))


                    total_sibling_state_names.append((empty_init_names, empty_init_names))


                parent_nodes_names = [node for is_p, node, is_name in zip(is_parent, nodes, name_indices) if is_p and is_name]

                # Update the parent nodes, children first, then siblings
                parent_nodes = [node for is_p, node, is_name in zip(is_parent, nodes, name_indices) if is_p and not is_name] \
                                + [node for has_s, node in zip(has_sibling, parent_nodes) if has_s]


                # Update the oov name index to token vocabs, children first, then siblings
                names_index2token_names = None

                if names_index2token is not None: 
                    names_index2token_names = [vocab for is_p, vocab, is_name in zip(is_parent, names_index2token, name_indices) if is_p and is_name]

                    names_index2token = [vocab for is_p, vocab, is_name in zip(is_parent, names_index2token, name_indices) if is_p and not is_name] \
                                            + [vocab for has_s, vocab in zip(has_sibling, names_index2token) if has_s]


                program_ids_names = [idx for is_p, idx, is_name in zip(is_parent, program_ids, name_indices) if is_p and is_name]


                # update the program ids (to keep track of declared names), children first, then siblings
                program_ids = [idx for is_p, idx, is_name in zip(is_parent, program_ids, name_indices) if is_p and not is_name] \
                            + [idx for has_s, idx in zip(has_sibling, program_ids) if has_s]


                sibling_path_offsets_names = [offset + [0] for is_p, offset, is_name in zip(is_parent, sibling_path_offsets, name_indices) if is_p and is_name]

                sibling_path_offsets = [offset + [0] for is_p, offset, is_name in zip(is_parent, sibling_path_offsets, name_indices) if is_p and not is_name] \
                                        + [offset[:-1] + [offset[-1] + 1] for has_s, offset in zip(has_sibling, sibling_path_offsets) if has_s]


                name_eval_nodes.add_nodes(total_parent_state_names, total_sibling_state_names, names_index2token_names, parent_nodes_names, program_ids_names, sibling_path_offsets_names)

                if len(parent_nodes) > 0:
                    self.decode_eval(total_parent_state, 
                                    total_sibling_state, 
                                    names_index2token, 
                                    name_eval_nodes, 
                                    temperature, 
                                    top_k, 
                                    top_p, 
                                    parent_nodes, 
                                    declared_names, 
                                    program_ids, 
                                    sibling_path_offsets, 
                                    iteration + 1)

            if not name_eval_nodes.is_empty():
                total_parent_state, total_sibling_state, names_index2token, parent_nodes, program_ids, sibling_path_offsets = name_eval_nodes.get_next()

                self.decode_eval(total_parent_state, 
                                total_sibling_state, 
                                names_index2token, 
                                name_eval_nodes, 
                                temperature, 
                                top_k, 
                                top_p, 
                                parent_nodes, 
                                declared_names, 
                                program_ids, 
                                sibling_path_offsets, 
                                iteration + 1)


        # If we are done, return the parent nodes (which contain the entire trees)
        return parent_nodes


    def is_declared(self, current_name_sib_path, decl_name_sib_path):
        for cur, decl in zip(current_name_sib_path, decl_name_sib_path):
            if cur > decl:
                return True
            if cur < decl:
                return False


class NameEvalNodes:
    def __init__(self, device) -> None:
        self.device = device
        self.parent_states = []
        self.sibling_states = []
        self.names_index2token = None
        self.parent_nodes = []
        self.program_ids = []
        self.sibling_path_offsets = []
        self.processing_names = False


    def add_nodes(self, parent_states, sibling_states, names_index2token, parent_nodes, program_ids, sibling_path_offsets):
        if names_index2token is not None:
            if self.names_index2token is None:
                self.names_index2token = list(names_index2token)
            else:
                self.names_index2token.extend(names_index2token)

        self.parent_nodes.extend(parent_nodes)
        self.program_ids.extend(program_ids)
        self.sibling_path_offsets.extend(sibling_path_offsets)


        if len(self.parent_states) == 0:
            self.parent_states.extend(parent_states)
            self.sibling_states.extend(sibling_states)

        else:
            for i in range(len(self.parent_states)):
                self.parent_states[i] = (torch.cat([self.parent_states[i][0], parent_states[i][0]]), torch.cat([self.parent_states[i][1], parent_states[i][1]])) 
                self.sibling_states[i] = (torch.cat([self.sibling_states[i][0], sibling_states[i][0]]), torch.cat([self.sibling_states[i][1], sibling_states[i][1]])) 

    def is_empty(self):
        return len(self.parent_nodes) == 0


    def sort(self):
        sorted_idxs = self.get_sorted_idxs()

        self.parent_nodes = np.array(self.parent_nodes)[sorted_idxs]
        self.program_ids = np.array(self.program_ids)[sorted_idxs]

        if self.names_index2token is not None:
            self.names_index2token = list(np.array(self.names_index2token)[sorted_idxs])

        for i in range(len(self.parent_states)):
            self.parent_states[i] = (self.parent_states[i][0][sorted_idxs], self.parent_states[i][1][sorted_idxs])
            
        # Sibling states dont have to be sorted, they are always empty for names
        

    def get_sorted_idxs(self):
        length = len(self.sibling_path_offsets)
        idxs = list(range(length))

        for i in range(length):
            for j in range(length - i - 1):
                for first, second in zip(self.sibling_path_offsets[j], self.sibling_path_offsets[j + 1]):
                    if first > second:
                        temp = self.sibling_path_offsets[j]
                        self.sibling_path_offsets[j] = self.sibling_path_offsets[j + 1]
                        self.sibling_path_offsets[j + 1] = temp

                        temp_idx = idxs[j]
                        idxs[j] = idxs[j + 1]
                        idxs[j + 1] = temp_idx
                        break
                    if second > first:
                        break

        return idxs



    def get_next(self):
        if not self.processing_names:
            self.processing_names = True
            self.sort()

        # get indices of first occurences of the program ids
        idxs = np.unique(self.program_ids, return_index=True)[1]

        next_program_ids = list(np.array(self.program_ids)[idxs])
        next_names_index2token = None

        if self.names_index2token is not None:
            next_names_index2token = list(np.array(self.names_index2token)[idxs])

        next_parent_nodes = list(np.array(self.parent_nodes)[idxs])
        next_sibling_path_offsets = list(np.array(self.sibling_path_offsets, dtype=object)[idxs])

        next_parent_states = []
        next_sibling_states = []

        for i in range(len(self.parent_states)):
            next_parent_states.append((self.parent_states[i][0][idxs], self.parent_states[i][1][idxs]))
            next_sibling_states.append((self.sibling_states[i][0][idxs], self.sibling_states[i][1][idxs]))

        self.remove_next(idxs)


        return next_parent_states, next_sibling_states, next_names_index2token, next_parent_nodes, next_program_ids, next_sibling_path_offsets


    def remove_next(self, idxs):
        self.sibling_path_offsets = list(np.delete(np.array(self.sibling_path_offsets, dtype='object'), idxs, axis=0))
        self.program_ids = list(np.delete(self.program_ids, idxs, axis=0))

        if self.names_index2token is not None:
            self.names_index2token = list(np.delete(np.array(self.names_index2token), idxs, axis=0))

        self.parent_nodes = list(np.delete(np.array(self.parent_nodes), idxs, axis=0))
        

        for i in range(len(self.parent_states)):
            self.parent_states[i] = (self.parent_states[i][0][np.setdiff1d(range(len(self.parent_states[i][0])), idxs)], self.parent_states[i][1][np.setdiff1d(range(len(self.parent_states[i][1])), idxs)]) # (np.delete(self.parent_states[i][0].cpu(), idxs).to(self.device), np.delete(self.parent_states[i][1].cpu(), idxs).to(self.device))
            self.sibling_states[i] = (self.sibling_states[i][0][np.setdiff1d(range(len(self.sibling_states[i][0])), idxs)], self.sibling_states[i][1][np.setdiff1d(range(len(self.sibling_states[i][1])), idxs)]) # (np.delete(self.sibling_states[i][0].cpu(), idxs).to(self.device), np.delete(self.sibling_states[i][1].cpu(), idxs).to(self.device))

        # print(len(self.parent_states), self.parent_states, self.parent_states[0][0].shape)



