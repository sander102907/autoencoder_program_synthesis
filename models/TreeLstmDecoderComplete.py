import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.TreeNode import Node

class TreeLstmDecoderComplete(nn.Module):
    def __init__(self, device, params, embedding_layers):
        super().__init__()
        
        self.device = device
        
        self.params = params
        self.latent_dim = params['LATENT_DIM']
        self.embedding_layers = embedding_layers
        self.leaf_lstms_sibling = nn.ModuleDict({})
        self.prediction_layers = nn.ModuleDict({})

        
        self.lstm_parent = nn.LSTMCell(params['EMBEDDING_DIM'], params['LATENT_DIM'])
        self.U_parent = nn.Linear(params['LATENT_DIM'], params['LATENT_DIM'], bias=False)
        self.depth_pred = nn.Linear(params['LATENT_DIM'], 1)
        
        self.lstm_sibling = nn.LSTMCell(params['EMBEDDING_DIM'], params['LATENT_DIM'])
        self.U_sibling = nn.Linear(params['LATENT_DIM'], params['LATENT_DIM'], bias=False)
        self.width_pred = nn.Linear(params['LATENT_DIM'], 1)

        # Leaf lstms and prediction layers
        for k, embedding_layer in embedding_layers.items():
            if not 'RES' in k:
                self.leaf_lstms_sibling[k] = nn.LSTMCell(params['LEAF_EMBEDDING_DIM'], params['HIDDEN_SIZE'])
            self.prediction_layers[k] = nn.Linear(params['LATENT_DIM'], params[f'{k}_VOCAB_SIZE'])

        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=-1)

        self.cross_ent_loss = nn.CrossEntropyLoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        
        self.offset_parent = nn.Linear(1, 1, bias=False)
        self.offset_sibling = nn.Linear(1, 1, bias=False)
        
        
    def forward(self, z, target=None):
        # We are training and we can do teacher forcing and batch processing
        if target is not None:      
            # print(target['features'][target['vocabs'] == 'LITERAL'])
            # print(target['features'][target['vocabs'] == 'RES'])      
            # num_nodes = len(target['node_order_bottomup'][target['node_order_bottomup'] == 0])
            # num_leafs = len(target['node_order_topdown']) - num_nodes
            
            # Initalize output
            # output = {
                    # 'predicted_labels': torch.empty(0, device=self.device),
                    # 'labels': torch.empty(0, device=self.device),
                    # 'predicted_labels_leaf': torch.empty(0, device=self.device),
                    # 'labels_leaf': torch.empty(0, device=self.device),
                    # 'predicted_labels': torch.zeros(sum(target['tree_sizes']), self.res_vocab_size, device=self.device),
                    # 'labels': torch.zeros(sum(target['tree_sizes']), device=self.device),
                    # 'predicted_labels_leaf': torch.zeros(sum(target['tree_sizes']), self.vocab_size, device=self.device),
                    # 'labels_leaf': torch.zeros(sum(target['tree_sizes']), device=self.device),
                    # 'predicted_has_siblings': torch.zeros(sum(target['tree_sizes']),  device=self.device),
                    # 'has_siblings': torch.zeros(sum(target['tree_sizes']), device=self.device),
                    # 'predicted_is_parent': torch.zeros(sum(target['tree_sizes']), device=self.device),
                    # 'is_parent': torch.zeros(sum(target['tree_sizes']), device=self.device)}

            # for k in self.embedding_layers.keys():
            #     output[f'{k}_predicted_labels'] = torch.empty(0, device=self.device)
            #     output[f'{k}_labels'] = torch.empty(0, device=self.device)

            loss = 0


            node_order = target['node_order_topdown']
            edge_order = target['edge_order_topdown']
            features = target['features']
            adj_list = target['adjacency_list']
            vocabs = target['vocabs']

            total_nodes = node_order.shape[0]

            # h and c states for every node in the batch for parent lstm
            h_p = torch.zeros(total_nodes, self.latent_dim, device=self.device)
            c_p = torch.zeros(total_nodes, self.latent_dim, device=self.device)
            
            #  h and c states for every node in the batch for sibling lstm
            h_s = torch.zeros(total_nodes, self.latent_dim, device=self.device)
            c_s = torch.zeros(total_nodes, self.latent_dim, device=self.device)

            for iteration in range(node_order.max() + 1):
                loss += self.decode_train(iteration, z, h_p, c_p, h_s, c_s, node_order, edge_order, features, adj_list, vocabs)
            

            return loss / total_nodes
        
        # We are evaluating and we cannot use training forcing and we generate tree by tree
        else:
            trees = []
            c_parent = torch.zeros(1, self.latent_dim, device=self.device)
            
            for index in range(z.shape[0]):
                trees.append(self.decode_eval((z[index].unsqueeze(0), c_parent), None))
                
            return trees

            
    def decode_train(self, iteration, z, h_p, c_p, h_s, c_s, node_order, edge_order, features, adj_list, vocabs):
        batch_size = z.shape[0]
        loss = 0
        
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
            h_parent, c_parent, h_prev_sibling, c_prev_sibling, is_parent, has_sibling, current_nodes_indices, vocabs_mask = \
                                                                            self.get_hidden_values(iteration, adj_list, edge_order, h_p, c_p, h_s, c_s,
                                                                                                   sibling_index, first_sibling_indices,
                                                                                                   current_indices if iteration > 0 else [],
                                                                                                   node_order, parent_indices, vocabs, z)
            
            h_pred = torch.tanh(self.U_parent(h_parent) + self.U_sibling(h_prev_sibling))

            # # Probability of the node having children
            # p_parent = self.sigmoid(self.depth_pred(h_pred))
            # # Probability of the node having successor children
            # p_sibling = self.sigmoid(self.width_pred(h_pred))

            loss += self.bce_loss(self.depth_pred(h_pred), is_parent)
            loss += self.bce_loss(self.width_pred(h_pred), has_sibling)

            # Get true label values
            label = features[current_nodes_indices].long()


            # Iterate over possible node types and predict labels for each node type
            for k, prediction_layer in self.prediction_layers.items():
                # Only do actual calculations when the amount of nodes for this node type > 0
                if len(h_pred[vocabs_mask == k]) > 0:
                    # Get label predictions
                    label_pred = prediction_layer(h_pred[vocabs_mask == k])

                    # Calculate cross entropy loss of label prediction
                    loss += self.cross_ent_loss((label_pred 
                        + self.offset_parent(is_parent[vocabs_mask == k]) 
                        + self.offset_sibling(has_sibling[vocabs_mask == k])), label[vocabs_mask == k].view(-1))

                    # predicted_labels[k] = self.softmax(
                    #     label_pred 
                    #     + self.offset_parent(is_parent[vocabs_mask == k]) 
                    #     + self.offset_sibling(has_sibling[vocabs_mask == k]))

                    # Calculate embedding of true label -> teacher forcing
                    if 'RES' in k:
                        embedding_dim = self.params['EMBEDDING_DIM']
                    else:
                        embedding_dim = self.params['LEAF_EMBEDDING_DIM']

                    emb_label = self.embedding_layers[k](label[vocabs_mask == k]).view(-1, embedding_dim)

                # If there are more than 0 elements processed for this node type
                # if emb_label.shape[0] > 0:
                    if 'RES' in k:
                        # Compute hidden and cell values of current nodes
                        h_parent, c_parent = self.lstm_parent(emb_label, (h_parent[vocabs_mask == k], c_parent[vocabs_mask == k]))
                        h_sibling, c_sibling = self.lstm_sibling(emb_label, (h_prev_sibling[vocabs_mask == k], c_prev_sibling[vocabs_mask == k]))

                        # Update the hidden and cell values matrices
                        h_p[current_nodes_indices[vocabs_mask == k]] = h_parent
                        c_p[current_nodes_indices[vocabs_mask == k]] = c_parent
                        
                        h_s[current_nodes_indices[vocabs_mask == k]] = h_sibling
                        c_s[current_nodes_indices[vocabs_mask == k]] = c_sibling

                    else:
                        h, c = self.leaf_lstms_sibling[k](
                            emb_label,
                            (h_prev_sibling[vocabs_mask == k],
                            c_prev_sibling[vocabs_mask == k]))

                        h_s[current_nodes_indices[vocabs_mask == k]] = h
                        c_s[current_nodes_indices[vocabs_mask == k]] = c
                


            # Label prediction from hidden state
            # label_pred = self.label_prediction(h_pred[is_parent_mask])

            # # Leaf label prediction from hidden state
            # leaf_label_pred = self.leaf_label_prediction(h_pred[~is_parent_mask])

                      
            # Node label prediction
            # predicted_label = self.softmax(
            #     label_pred + self.offset_parent(is_parent[is_parent_mask]) + self.offset_sibling(has_sibling[is_parent_mask]))

            # # Leaf label prediction
            # predicted_label_leaf = self.softmax(
            #     leaf_label_pred + self.offset_parent(is_parent[~is_parent_mask]) + self.offset_sibling(has_sibling[~is_parent_mask]))


            # Get embedding of node label
            # emb_label = self.res_embedding(label[is_parent_mask]).view(-1, self.res_embedding_dim)

            # # Get embedding of leaf label
            # emb_leaf_label = self.embedding(label[~is_parent_mask]).view(-1, self.embedding_dim)

            # if emb_label.shape[0] > 0:
            #     # Compute hidden and cell values of current nodes
            #     h_parent, c_parent = self.lstm_parent(emb_label, (h_parent[is_parent_mask], c_parent[is_parent_mask]))
            #     h_sibling, c_sibling = self.lstm_sibling(emb_label, (h_prev_sibling[is_parent_mask], c_prev_sibling[is_parent_mask]))

            #     # Update the hidden and cell values matrices
            #     h_p[current_nodes_indices[is_parent_mask]] = h_parent
            #     c_p[current_nodes_indices[is_parent_mask]] = c_parent
                
            #     h_s[current_nodes_indices[is_parent_mask]] = h_sibling
            #     c_s[current_nodes_indices[is_parent_mask]] = c_sibling
            
            # if emb_leaf_label.shape[0] > 0:
            #     h_sibling_leaf, c_sibling_leaf = self.leaf_lstm_sibling(emb_leaf_label, (h_prev_sibling[~is_parent_mask], c_prev_sibling[~is_parent_mask]))

            #     h_s[current_nodes_indices[~is_parent_mask]] = h_sibling_leaf
            #     c_s[current_nodes_indices[~is_parent_mask]] = c_sibling_leaf
                
            
            
                        
            # For computing loss, save output (predictions and true values)
            # if predicted_label.shape[0] > 0:


            return loss
            # for k in self.embedding_layers.keys():
            #     if len(current_nodes_indices[vocabs_mask == k]) > 0:

            #         output[f'{k}_predicted_labels'] = torch.cat([output[f'{k}_predicted_labels'] , predicted_labels[k]])
            #         output[f'{k}_labels'] = torch.cat([output[f'{k}_labels'], features[current_nodes_indices[vocabs_mask == k]].view(-1)])

                    # if k == 'LITERAL':
                    #     print(output[f'{k}_labels'])
                    #     print(torch.argmax(output[f'{k}_predicted_labels'], dim=-1))
                        # print(output[f'{k}_predicted_labels'].shape, output[f'{k}_labels'].shape)

            # output['predicted_labels'] = torch.cat([output['predicted_labels'], predicted_label])
            # output['labels'] = torch.cat([output['labels'], features[current_nodes_indices[is_parent_mask]].view(-1)])

            # # if predicted_label_leaf.shape[0] > 0:
            # output['predicted_labels_leaf'] = torch.cat([output['predicted_labels_leaf'], predicted_label_leaf])
            # output['labels_leaf'] = torch.cat([output['labels_leaf'], features[current_nodes_indices[~is_parent_mask]].view(-1)])
            
            # output['predicted_labels'][current_nodes_indices[is_parent_mask]] = predicted_label
            # output['labels'][current_nodes_indices[is_parent_mask]] = features[current_nodes_indices[is_parent_mask]].view(-1)
            # output['predicted_labels_leaf'][current_nodes_indices[~is_parent_mask]] = predicted_label_leaf
            # output['labels_leaf'][current_nodes_indices[~is_parent_mask]] = features[current_nodes_indices[~is_parent_mask]].view(-1)
            # output['predicted_has_siblings'][current_nodes_indices] = p_sibling.view(-1)
            # output['has_siblings'][current_nodes_indices] = has_sibling.view(-1)
            # output['predicted_is_parent'][current_nodes_indices] = p_parent.view(-1)
            # output['is_parent'][current_nodes_indices] = is_parent.view(-1)

    def get_hidden_values(self, iteration, adj_list, edge_order, h_p, c_p, h_s, c_s, sibling_index,
                          first_sibling_indices, current_indices, node_order, parent_indices, vocabs, z):
        
        batch_size = z.shape[0]
        
        # At sibling index 0, there should not be any previous siblings
        if sibling_index == 0:
            num_first_siblings = len(first_sibling_indices)
            # Initialize to hidden, cell of siblings to zero
            h_prev_sibling = torch.zeros(num_first_siblings, self.latent_dim, device=self.device)
            c_prev_sibling = torch.zeros(num_first_siblings, self.latent_dim, device=self.device)

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
            c_parent = torch.zeros(batch_size, self.latent_dim, device=self.device)    
        else:
            h_parent = h_p[parent_indices_siblings, :]
            c_parent = c_p[parent_indices_siblings, :]

        adj_list_curr = adj_list[edge_order == iteration, :]
        sib = list(first_sibling_indices) + [len(parent_indices)]
        vocabs_mask = np.atleast_1d(vocabs[current_nodes_indices.cpu()])

        is_parent = torch.tensor([[1.] if index in adj_list_curr[:, 0] else [0.] for index in current_nodes_indices], device=self.device)
        has_sibling = torch.tensor([[1.] if j-i -1 > sibling_index else [0.] for i,  j in zip(sib[:-1], sib[1:]) if j-i > sibling_index], device=self.device)
        
        
        return h_parent, c_parent, h_prev_sibling, c_prev_sibling, is_parent, has_sibling, current_nodes_indices, vocabs_mask
    
    
    def decode_eval(self, parent_state, sibling_state, parent_node=None):        
        h_parent, c_parent = parent_state
            
        if sibling_state is not None:
            h_prev_sibling, c_prev_sibling = sibling_state
        else:
            # Initialize to hidden, cell of siblings to zero
            h_prev_sibling = torch.zeros(1, self.latent_dim, device=self.device)
            c_prev_sibling = torch.zeros(1, self.latent_dim, device=self.device)
            
            
        h_pred = torch.tanh(self.U_parent(h_parent) + self.U_sibling(h_prev_sibling))

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

        if is_parent:
            # Label prediction from hidden state
            label_pred = self.label_prediction(h_pred)

        else:
            # Leaf label prediction from hidden state
            label_pred = self.leaf_label_prediction(h_pred)
            

        # Node label prediction
        predicted_label = self.softmax(label_pred + self.offset_parent(is_parent) + self.offset_sibling(has_sibling))
        
        # Build tree: Add node to tree
        if parent_node is None:
            node = Node(torch.argmax(predicted_label, dim=-1).item(), is_reserved=True, parent=None)
        else:
            node = Node(torch.argmax(predicted_label, dim=-1).item(), is_reserved=True if is_parent else False, parent=parent_node)
                    
        if is_parent:
            # Take argmax of predicted label and transform to onehot
            # predicted_label = F.one_hot(torch.argmax(predicted_label, dim=-1), self.vocab_size).float().view(-1, self.vocab_size).to(self.device)
            emb_label = self.res_embedding(torch.argmax(predicted_label, dim=-1)).view(-1, self.res_embedding_dim)
        else:
            emb_label = self.embedding(torch.argmax(predicted_label, dim=-1)).view(-1, self.embedding_dim)
                    
        # If we predict a next sibling
        if has_sibling:
            
            if is_parent:
                # Calculate next hidden sibling state
                sibling_state = self.lstm_sibling(emb_label, (h_prev_sibling, c_prev_sibling))
            else:
                # Calculate next hidden sibling state
                sibling_state = self.leaf_lstm_sibling(emb_label, (h_prev_sibling, c_prev_sibling))
            
            # Pass the same parent state, but updated sibling state
            self.decode_eval(parent_state, sibling_state, parent_node)
            
        # We set the created node as the parent node
        parent_node = node
        
        # If we predict we are a parent, continue with children
        if is_parent:
            # update parent state and parent_node of tree
            parent_state = self.lstm_parent(emb_label, parent_state)
            
            # Pass new parent state and no sibling state as we start with the first sibling
            self.decode_eval(parent_state, None, parent_node)
        
        
        # If we are done, return the root node (which contains the entire tree)
        return parent_node
        
        
