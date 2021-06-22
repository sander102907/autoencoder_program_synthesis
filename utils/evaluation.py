from utils.TreeNode import Node
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import rand_score, adjusted_rand_score
from anytree.search import findall
from anytree.exporter import JsonExporter
from cpp_ast_parser.AST_to_code import AstToCodeParser
from cpp_ast_parser.utils import add_includes_usings
import os


class Evaluator:
    def __init__(self, vocabulary, adjusted_rand=False):
        self.bleu_hypotheses = []
        self.bleu_references = []

        self.rand_hypotheses = []
        self.rand_references = []

        self.total_bleu_nodes_ref = 0
        self.total_rand_nodes_ref = 0

        self.total_bleu_nodes_hyp = 0
        self.total_rand_nodes_hyp = 0

        self.vocabulary = vocabulary
        self.adjusted_rand = adjusted_rand

    # def calc_total_score(self):
    #     self._calc_bleu_score()
    #     self._calc_rand_score()

    #     total_nodes_ref = self.total_bleu_nodes_ref + self.total_rand_nodes_ref
    #     total_nodes_hyp = self.total_bleu_nodes_hyp + self.total_rand_nodes_hyp

    #     perc_nodes_bleu_ref = self.total_bleu_nodes_ref / total_nodes_ref
    #     perc_nodes_rand_ref = self.total_rand_nodes_ref / total_nodes_ref

    #     perc_nodes_bleu_hyp = self.total_bleu_nodes_hyp / total_nodes_hyp
    #     perc_nodes_rand_hyp = self.total_rand_nodes_hyp / total_nodes_hyp

    #     self.total_score_ref = self.bleu_4 * perc_nodes_bleu_ref + self.rand_score * perc_nodes_rand_ref
    #     self.total_score_hyp = self.bleu_4 * perc_nodes_bleu_hyp + self.rand_score * perc_nodes_rand_hyp

    def calc_bleu_score(self):
        self.bleu_4 = corpus_bleu(self.bleu_references, self.bleu_hypotheses)
        self.bleu_3 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1/3, 1/3, 1/3, 0))
        self.bleu_2 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1/2, 1/2, 0, 0))
        self.bleu_1 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1, 0, 0, 0))

        return {
            'bleu_1': self.bleu_1,
            'bleu_2': self.bleu_2,
            'bleu_3': self.bleu_3,
            'bleu_4': self.bleu_4
        }


    def reconstructions_to_file(self, reconstructions, folder):
        parser = AstToCodeParser('output/')
        parser.load_vocabs_from_dicts(self.vocabulary.token2index)
        os.makedirs(os.path.join('output', folder), exist_ok=True)

        imports = ['using namespace std;', '#include <vector>', '#include <iostream>', '#include <string>',
                  '#include <cstring>', '#include <queue>', '#include <stdio.h>', '#include <math.h>', '#include <map>']

        exporter = JsonExporter()

        for idx, tree in enumerate(reconstructions):
            self._add_main_to_reconstruction(tree)
            program_path = os.path.join('output', folder, f'{idx}.cpp')
            with open(program_path, 'w') as f:
                try:
                    for child in tree.children:
                        f.write(parser.parse_node(child))
                except Exception as e:
                    pass

            with open(program_path.replace('cpp', 'json'), 'w') as f:
                f.write(exporter.export(tree))

            add_includes_usings(program_path, imports)


    def _add_main_to_reconstruction(self, reconstruction):
        """
            Set the name of the first function declaration that has a body (compount statement)
            to the name "main"
        """
        func_decl_names = findall(reconstruction, filter_=lambda node: node.parent is not None 
                                   and node.parent.token == self.vocabulary.token2index['RES']['NAME'] 
                                   and node.parent.parent is not None 
                                   and node.parent.parent.token == self.vocabulary.token2index['RES']['FUNCTION_DECL']
                                   and node.parent.parent.children[-1].token == self.vocabulary.token2index['RES']['COMPOUND_STMT'])


        if len(func_decl_names) > 0:
            func_decl_names[0].token = 'main'


    def _calc_rand_score(self):
        if self.adjusted_rand:
            rand_scores = [adjusted_rand_score(hyp, ref) for hyp, ref in zip(self.rand_hypotheses, self.rand_references)]
        else:
            rand_scores = [rand_score(hyp, ref) for hyp, ref in zip(self.rand_hypotheses, self.rand_references)]
            
        self.rand_score = sum(rand_scores) / len(rand_scores)


    def add_eval_hypotheses(self, batch):
        tree_offset = 0

        for tree_size in batch['tree_sizes']:
            ast = self._build_ast(batch['adjacency_list'], batch['features'], batch['vocabs'], tree_offset)
            bleu = self._ast_to_eval_tokens(ast) 

            self.bleu_hypotheses.append(bleu)
            # self.rand_hypotheses.append(rand)

            tree_offset += tree_size
            self.total_bleu_nodes_hyp += len(bleu)
            # self.total_rand_nodes_hyp += len(rand)


    def add_eval_references(self, asts):
        references = []

        for ast in asts:
            bleu = self._ast_to_eval_tokens(ast)

            self.bleu_references.append([bleu])
            # self.rand_references.append(rand)

            self.total_bleu_nodes_ref += len(bleu)
            # self.total_bleu_nodes_hyp += len(rand)

        return references


    def _build_ast(self, adj_list, features, vocabs, index=0, parent_node=None):
        node = Node(features[index].item(), is_reserved=vocabs[index] == 'RES', parent=parent_node)
        children = adj_list[adj_list[:, 0] == index][:, 1]

        for child in children:
            self._build_ast(adj_list, features, vocabs, child, node)

        return node


    def _ast_to_eval_tokens(self, ast):
        reserved_eval_tokens = ['IF_STMT', 'WHILE_STMT', 'CONTINUE_STMT', 'CXX_FOR_RANGE_STMT', 'FOR_STMT',
        'RETURN_STMT', 'SWITCH_STMT', 'CASE_STMT', 'NULL_STMT', 'BREAK_STMT', 'DO_STMT', 'TEMPLATE_DECL',
        'STRUCT_DECL', 'POINTER', 'CONST_QUALIFIED', 'LVALUEREFERENCE', 'RVALUEREFERENCE', 'CONST', 
        'PAREN_EXPR', 'GNU_NULL_EXPR', 'CXX_NULL_PTR_LITERAL_EXPR', 'ARRAY_SUBSCRIPT_EXPR', 'TYPEDEF_DECL', 
        'CLASS_DECL', 'STRUCT_DECL', 'INIT_LIST_EXPR', 'PACK_EXPANSION_EXPR', 'CXX_THIS_EXPR', 'CXX_TRY_STMT', 
        'CXX_CATCH_STMT', 'CXX_STATIC_CAST_EXPR', 'GOTO_STMT', 'LABEL_STMT', 'CONSTRUCTOR_INITIALIZER', 'CXX_FUNCTIONAL_CAST_EXPR'
        'TYPE_CALL_EXPR']

        reserved_eval_idxs = []


        for idx, token in self.vocabulary.index2token['RES'].items():
            if token in reserved_eval_tokens or 'OPERATOR' in token:
                reserved_eval_idxs.append(idx)


        tokens = []
        # eval_name_tokens = []

        # if self._is_name_token(ast):
        #     eval_name_tokens.append(ast.token)
        if len(ast.children) == 0 or ast.token in reserved_eval_idxs:
            if 'NAME_' in str(ast.token):
                tokens.append(int(ast.token.replace('NAME_', '')))
            else:
                tokens.append(self._loc_to_glob_token(ast.token, self._get_vocab_type(ast)))

        for child in ast.children:
            tokens.extend(self._ast_to_eval_tokens(child))
            # eval_tokens.extend(tokens)
            # eval_name_tokens.extend(name_tokens)
                    
        return tokens #, eval_name_tokens


    def _is_name_token(self, node):
        if node.parent is not None and len(node.children) == 0:
            parent_token = self.vocabulary.index2token['RES'][node.parent.token]
            if not ('LITERAL' in parent_token or parent_token == 'TYPE' or parent_token == 'REF_BUILTIN'):
                return True

        return False


    def _loc_to_glob_token(self, token_idx, vocab_type):
        token = self.vocabulary.index2token[vocab_type][token_idx]
        return self.vocabulary.token2index['ALL'][token]

    def _get_vocab_type(self, node):
        parent_label = self.vocabulary.index2token['RES'][node.parent.token]

        if node.res:
            return 'RES'
        if 'LITERAL' in parent_label:
            return 'LITERAL'
        elif 'REF_BUILTIN' == parent_label:
            return 'NAME_BUILTIN'
        elif 'TYPE' == parent_label:
            return 'TYPE'
        else:
            return 'NAME'
        
        

