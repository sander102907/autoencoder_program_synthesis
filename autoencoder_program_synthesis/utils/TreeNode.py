from anytree import NodeMixin

"""
Custom tree node for the anytree module to contain
- pred: predicted label
- parent: the parent node
- children: the children nodes
- decl_line: The line number in the source code that contains the declaration of the ast item (e.g. declaration of reference of variable)
"""

class Node(NodeMixin): 
    def __init__(self, token, is_reserved, parent=None, children=None, decl_line=None):
        super(Node, self).__init__()
        self.token = token
        self.res = is_reserved
        self.parent = parent
        
        if decl_line:
            self.decl_line = decl_line

        if children:
            self.children = children
            
    def __str__(self):
        return str(self.token)