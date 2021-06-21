import numpy as np
import torch

def calculate_evaluation_orders_topdown(adj_list, adj_list_sib, tree_size):
    """
    This is a copy of TreeLSTM calculate_evaluation_orders method, but then reversed 
    to get the orders from top to bottom, instead of from bottom to top.
    (https://github.com/unbounce/pytorch-tree-lstm/blob/master/treelstm/util.py)
    This is required for a tree decoder as it works top-down.
    
    Calculates the node_order and edge_order from a tree adjacency_list and the tree_size
    from the top of the tree to the bottom.
    
    The Tree LSTM decoder requires node and edge order to calculate with multiple nodes in
    tree batches instead of the slow recursive approach. These orders are pre-calculated
    to speed up the process.
    """
    
    adj_list = np.array(adj_list)
    adj_list_sib = np.array(adj_list_sib)
    node_ids = np.arange(tree_size, dtype=int)

    node_order = np.zeros(tree_size, dtype=int)
    unevaluated_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adj_list[:, 0]
    child_nodes = adj_list[:, 1]

    prev_sib_nodes = adj_list_sib[:, 0]
    next_sib_nodes = adj_list_sib[:, 1]


    n = 0

    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        unevaluated_mask_sib = unevaluated_nodes[prev_sib_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]
        unready_next_sib = next_sib_nodes[unevaluated_mask_sib]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated child nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children) & ~np.isin(node_ids, unready_next_sib)

        # Set the node order
        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    add_sibling_order = np.zeros(len(parent_nodes), dtype=int)

    p_nodes_seen = {k : 0 for k in np.unique(parent_nodes)}

    for idx, val in enumerate(parent_nodes):
        to_add = p_nodes_seen[val]
        add_sibling_order[idx] += to_add
        p_nodes_seen[val] += 1


    edge_order = node_order[parent_nodes] + add_sibling_order
    edge_order_sib = node_order[prev_sib_nodes]

    edge_order_sib[edge_order_sib == 0] = node_order[adj_list_sib[adj_list_sib[:, 0] == 0, 1]] - 1

    return node_order, edge_order, edge_order_sib


def calculate_evaluation_orders_topdown_old(adj_list, tree_size):
    """
    This is a copy of TreeLSTM calculate_evaluation_orders method, but then reversed 
    to get the orders from top to bottom, instead of from bottom to top.
    (https://github.com/unbounce/pytorch-tree-lstm/blob/master/treelstm/util.py)
    This is required for a tree decoder as it works top-down.
    
    Calculates the node_order and edge_order from a tree adjacency_list and the tree_size
    from the top of the tree to the bottom.
    
    The Tree LSTM decoder requires node and edge order to calculate with multiple nodes in
    tree batches instead of the slow recursive approach. These orders are pre-calculated
    to speed up the process.
    """
    
    adj_list = np.array(adj_list)
    node_ids = np.arange(tree_size, dtype=int)

    node_order = np.zeros(tree_size, dtype=int)
    unevaluated_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adj_list[:, 0]
    child_nodes = adj_list[:, 1]

    n = 0

    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated child nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children)
        # print(tree_size, unevaluated_mask, unready_children, nodes_to_evaluate)
        # Set the node order
        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1


    edge_order = node_order[parent_nodes]

    return node_order, edge_order


def batch_tree_input(batch):
    """
    This is a copy of TreeLSTM batch_tree_input method, but then expanded to include both bottomup and topdown
    node and edge orders.
    (https://github.com/unbounce/pytorch-tree-lstm/blob/master/treelstm/util.py)
    
    Combines a batch of tree dictionaries into a single batched dictionary for use by the TreeLSTM encoding and decoding models.
    
    batch - list of dicts with keys ('features', 'node_order_bottomup', 'node_order_topdown', 'edge_order_bottomup', 'edge_order_topdown', 'adjacency_list')
    returns a dict with keys ('features', 'node_order_bottomup', 'node_order_topdown', 'edge_order_bottomup', 'edge_order_topdown', 'adjacency_list', 'tree_sizes')
    """
    
    tree_sizes = [b['features'].shape[0] for b in batch]

    batched_features = torch.cat([b['features'] for b in batch])
    batched_features_combined = torch.cat([b['features_combined'] for b in batch])
    batched_node_order_bottomup = torch.cat([b['node_order_bottomup'] for b in batch])
    batched_node_order_topdown = torch.cat([b['node_order_topdown'] for b in batch])
    batched_edge_order_bottomup = torch.cat([b['edge_order_bottomup'] for b in batch])
    batched_edge_order_topdown = torch.cat([b['edge_order_topdown'] for b in batch])
    batched_edge_order_topdown_sib = torch.cat([b['edge_order_topdown_sib'] for b in batch])
    batched_vocabs = np.concatenate([b['vocabs'] for b in batch])
    batched_declared_names = [b['declared_names'] for b in batch]

    batched_adjacency_list = []
    batched_adjacency_list_sib = []
    offset = 0
    for n, b in zip(tree_sizes, batch):
        batched_adjacency_list.append(b['adjacency_list'] + offset)
        batched_adjacency_list_sib.append(b['adjacency_list_sib'] + offset)
        offset += n
    batched_adjacency_list = torch.cat(batched_adjacency_list)
    batched_adjacency_list_sib = torch.cat(batched_adjacency_list_sib)

    tree_indices = torch.cat([torch.full((tree_sizes[idx],), fill_value=idx) for idx in range(len(batch))])

    return {
        'features': batched_features,
        'features_combined': batched_features_combined,
        'node_order_bottomup': batched_node_order_bottomup,
        'node_order_topdown': batched_node_order_topdown,
        'edge_order_bottomup': batched_edge_order_bottomup,
        'edge_order_topdown': batched_edge_order_topdown,
        'edge_order_topdown_sib': batched_edge_order_topdown_sib,
        'adjacency_list': batched_adjacency_list,
        'adjacency_list_sib': batched_adjacency_list_sib,
        'tree_sizes': tree_sizes,
        'vocabs': batched_vocabs,
        'tree_indices': tree_indices,
        'declared_names': batched_declared_names
    }
