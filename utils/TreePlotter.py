from anytree.exporter import UniqueDotExporter
from TreeConverter import Node
import re

"""
Utility to plot n_ary and binary trees to png photo.
"""
class TreePlotter:
    @staticmethod
    def plot_tree(root, file_path, label_dict=None, res_label_dict=None, binary=False):
        if binary:
            root = TreePlotter.__binary_tree_to_plot_format(root)
            
        def get_label(node):
#             if node.res and res_label_dict is not None:
#                 print(f'label={res_label_dict[node.token]}')
#                 print(re.findall(r'^[a-zA-Z0-9=_]*', f'label={res_label_dict[node.token]}')[0].replace("_", "x"))
#                 return re.findall(r'^[a-zA-Z0-9=]*', f'label={res_label_dict[node.token]}')[0].replace("_", "x")
#                 return f'label={res_label_dict[node.token]}'
#             elif not node.res and label_dict is not None:
# #                 print(f'label={label_dict[node.token]}')
# #                 print(re.findall(r'^[a-zA-Z0-9=]*', f'label={label_dict[node.token]}')[0])
#                 print(re.findall(r'^[a-zA-Z0-9=]*', f'label={label_dict[node.token]}')[0])
#                 return re.findall(r'^[a-z0-9=]*', f'label={label_dict[node.token]}')[0]
#             else:
            return f'label=x'

        UniqueDotExporter(root,
                         nodeattrfunc=get_label #lambda n: f'label="{get_label(n)}" shape={"ellipse" if n.res else "box"}'
                         ).to_picture(file_path)
        
    @staticmethod
    def plot_predicted_tree(root, file_path):
        UniqueDotExporter(root,
                         nodeattrfunc=lambda n: f'label="{n.pred}"'
                         ).to_picture(file_path)

    @staticmethod
    def __binary_tree_to_plot_format(node, parent_node=None):
        new_node = Node(node.token, node.res, parent=parent_node)
        if node.left_child:
            TreePlotter.__binary_tree_to_plot_format(node.left_child, new_node)
        if node.right_child:
            TreePlotter.__binary_tree_to_plot_format(node.right_child, new_node)
        return new_node    