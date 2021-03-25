from anytree import NodeMixin

"""
Custom tree node for the anytree module to contain
- pred: predicted label
- parent: the parent node
- children: the children nodes
"""

class Node(NodeMixin): 
    def __init__(self, predicted_label, parent=None, children=None):
        super(Node, self).__init__()
        self.pred = predicted_label
        self.parent = parent
        if children:
            self.children = children
            
    def __str__(self):
        return str(self.pred)