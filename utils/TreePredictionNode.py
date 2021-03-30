from anytree import NodeMixin

"""
Custom tree node for the anytree module to contain
- pred: predicted label
- parent: the parent node
- children: the children nodes
"""

class Node(NodeMixin): 
    def __init__(self, predicted_label, parent=None, children=None):
        super().__init__()
        self.pred = predicted_label
        self.parent = parent
        if children:
            self.children = children
            
    def get_hidden_prev_sibling(self):
        siblings = self.parent.children
        if siblings.index(self) > 0:
            prev_sibling = siblings[siblings.index(self) - 1]
            return (prev_sibling.hidden, prev_sibling.cell)
        else:
            return None
        
            
    def __str__(self):
        return str(self.pred)