from anytree.exporter import UniqueDotExporter
from TreeConverter import Node

"""
Utility to plot n_ary and binary trees to png photo.
"""
class TreePlotter:
    @staticmethod
    def plot_tree(root, file_path, binary=False):
        if binary:
            root = TreePlotter.__binary_tree_to_plot_format(root)

        UniqueDotExporter(root,
                         nodeattrfunc=lambda n: f'label="{n.token}" shape={"ellipse" if n.res else "box"}'
                         ).to_picture(file_path)

    @staticmethod
    def __binary_tree_to_plot_format(node, parent_node=None):
        new_node = Node(node.token, node.res, parent=parent_node)
        if node.left_child:
            TreePlotter.__binary_tree_to_plot_format(node.left_child, new_node)
        if node.right_child:
            TreePlotter.__binary_tree_to_plot_format(node.right_child, new_node)
        return new_node    