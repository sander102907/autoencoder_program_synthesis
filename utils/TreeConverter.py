from anytree import NodeMixin

"""
Utility that convert an N-ary tree into a binary tree
and vice versa.

The N-ary to binary conversion is in the Left-Child Right-Sibling representation
"""
class TreeConverter:
    @staticmethod
    def to_binary(root):
        """Encodes an n-ary tree to a binary tree.
        
        :type root: Node
        :rtype: BinaryNode
        """
        def to_binary_helper(root, parent, index):
            if not root:
                return None
            node = BinaryNode(root.token, root.res)
            if index+1 < len(parent.children):
                node.right_child = to_binary_helper(parent.children[index+1], parent, index+1)
            if root.children:
                node.left_child = to_binary_helper(root.children[0], root, 0)
            return node

        if not root:
            return None
        node = BinaryNode(root.token, root.res)
        if root.children:
            node.left_child = to_binary_helper(root.children[0], root, 0)
        return node

    @staticmethod
    def to_n_arry(root):
        """Decodes your binary tree to an n-ary tree.
        
        :type root: BinaryNode
        :rtype: Node
        """
        def to_n_arry_helper(root, parent):
            if not root:
                return
            node = Node(root.token, root.res, parent=parent)
            to_n_arry_helper(root.left_child, node)
            to_n_arry_helper(root.right_child, parent)

        if not root:
            return None
        node = Node(root.token, root.res)
        to_n_arry_helper(root.left_child, node)
        return node


# Definition of an binary tree node.
class BinaryNode(NodeMixin):
    def __init__(self, token, is_reserved):
        super(BinaryNode, self).__init__()
        self.token = token
        self.res = is_reserved
        self.left_child = None
        self.right_child = None
        
    def __str__(self):
        return str(self.token)

# Definition of an n-ary tree node
class Node(NodeMixin): 
    def __init__(self, token, is_reserved, parent=None, children=None):
        super(Node, self).__init__()
        self.token = token
        self.res = is_reserved
        self.parent = parent
        if children:
            self.children = children
            
    def __str__(self):
        return str(self.token)