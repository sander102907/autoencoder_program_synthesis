from torch._C import Value
from utils.TreeNode import Node
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import rand_score, adjusted_rand_score
from anytree.search import findall
from anytree.exporter import JsonExporter
from cpp_ast_parser.AST_to_code import AstToCodeParser
from cpp_ast_parser.utils import add_includes_usings
import os
import subprocess
from threading import Timer
from tqdm import tqdm
from sctokenizer import CppTokenizer, TokenType
from copy import deepcopy


class Seq2SeqEvaluator:
    def __init__(self, vocabulary):
        self.bleu_hypotheses = []
        self.bleu_references = []
        self.ids = []

        self.vocabulary = vocabulary
        self.pad_idx = vocabulary.token2index['ALL']['<pad>']
        self.eos_idx = vocabulary.token2index['ALL']['<eos>']

    def calc_bleu_score(self, individual_scores=False):
        self.bleu_4 = corpus_bleu(self.bleu_references, self.bleu_hypotheses)
        self.bleu_3 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1/3, 1/3, 1/3, 0))
        self.bleu_2 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1/2, 1/2, 0, 0))
        self.bleu_1 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1, 0, 0, 0))

        
        if individual_scores:
            import pandas as pd

            bleu_scores = []
            for ref, hyp, id in zip(self.bleu_references, self.bleu_hypotheses, self.ids):
                bleu_4 = corpus_bleu([ref], [hyp])
                bleu_scores.append({'id': id, 'bleu_4': bleu_4})

            folder = os.path.join('output', 'bleu_scores')
            os.makedirs(folder, exist_ok=True)

            pd.DataFrame(bleu_scores).sort_values('bleu_4', ascending=False).to_csv(os.path.join(folder, 'bleu_scores_s2s.csv'), index=False)
            
        return {
            'bleu_1': self.bleu_1,
            'bleu_2': self.bleu_2,
            'bleu_3': self.bleu_3,
            'bleu_4': self.bleu_4
        }

    def calc_perc_compiles(self, folder, fix_errors=False):
        code_folder = os.path.join('output', folder, 'code')
        compile_folder = os.path.join('output', folder, 'compiled')
        os.makedirs(compile_folder, exist_ok=True)

        print('INFO - Experiment - Calculating percentage compiles..')

        subprocess.run('for entry in ' +  f'"{code_folder}"/*' + """
            do
            """ + f"{'#' if not fix_errors else ''} clang-tidy $entry -fix-errors" + """
            : ${entry/cpp/out} 
            g++ $entry -o  ${_/code/compiled}
            done""", shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return len(os.listdir(compile_folder)) / len(os.listdir(code_folder))


    def add_eval_hypotheses(self, orig_programs, ids):
        for idx, program in zip(ids, orig_programs):
            self.bleu_hypotheses.append([token for token in program if token != self.pad_idx and token != self.eos_idx])
            self.ids.append(idx)


    def add_eval_references(self, reconstructions):
        for program in reconstructions:
            self.bleu_references.append([[token.item() for token in program if token.item() != self.pad_idx and token.item() != self.eos_idx]])


    def reconstructions_to_file(self, reconstructions, folder, ids):
        code_folder = os.path.join('output', folder, 'code')
        os.makedirs(code_folder, exist_ok=True)

        imports = ['using namespace std;', '#include <vector>', '#include <iostream>', '#include <string>',
            '#include <cstring>', '#include <queue>', '#include <stdio.h>', '#include <math.h>', '#include <map>', '#include <set>', '#include <stack>']

        for idx, program in zip(ids, reconstructions):
            program_path = os.path.join(code_folder, f'{idx}.cpp')

            with open(program_path, 'w') as f:
                f.write(' '.join([self.vocabulary.index2token['ALL'][x.item()] for x in program if x.item() != self.pad_idx and x.item() != self.eos_idx]))


            add_includes_usings(program_path, imports)

        self.programs = []


    def generations_to_file(self, generations, folder):
        generations_folder = os.path.join('output', folder, 'generations')
        os.makedirs(generations_folder, exist_ok=True)

        imports = ['using namespace std;', '#include <vector>', '#include <iostream>', '#include <string>',
            '#include <cstring>', '#include <queue>', '#include <stdio.h>', '#include <math.h>', '#include <map>', '#include <set>', '#include <stack>']

        for program in generations:
            idx = len(os.listdir(generations_folder))
            program_path = os.path.join(generations_folder, f'{idx}.cpp')

            with open(program_path, 'w') as f:
                f.write(' '.join([self.vocabulary.index2token['ALL'][x.item()] for x in program if x.item() != self.pad_idx and x.item() != self.eos_idx]))


            add_includes_usings(program_path, imports)


class Tree2TreeEvaluator:
    def __init__(self, vocabulary, adjusted_rand=False):
        self.bleu_hypotheses = []
        self.bleu_references = []

        self.rand_hypotheses = []
        self.rand_references = []

        self.ids = []

        self.total_bleu_nodes_ref = 0
        self.total_rand_nodes_ref = 0

        self.total_bleu_nodes_hyp = 0
        self.total_rand_nodes_hyp = 0

        self.vocabulary = vocabulary
        self.adjusted_rand = adjusted_rand

        self.tokenizer = CppTokenizer()
        self.parser = AstToCodeParser('output/')
        self.parser.load_vocabs_from_dicts(self.vocabulary.token2index)

        self.programs = []

    def calc_bleu_score(self, individual_scores=False):
        self.bleu_4 = corpus_bleu(self.bleu_references, self.bleu_hypotheses)
        self.bleu_3 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1/3, 1/3, 1/3, 0))
        self.bleu_2 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1/2, 1/2, 0, 0))
        self.bleu_1 = corpus_bleu(self.bleu_references, self.bleu_hypotheses, weights=(1, 0, 0, 0))

        
        if individual_scores:
            import pandas as pd

            bleu_scores = []
            for ref, hyp, id in zip(self.bleu_references, self.bleu_hypotheses, self.ids):
                bleu_4 = corpus_bleu([ref], [hyp])
                bleu_scores.append({'id': id, 'bleu_4': bleu_4})

            folder = os.path.join('output', 'bleu_scores')
            os.makedirs(folder, exist_ok=True)

            pd.DataFrame(bleu_scores).sort_values('bleu_4', ascending=False).to_csv(os.path.join(folder, 'bleu_scores_t2t.csv'), index=False)
            
            
        return {
            'bleu_1': self.bleu_1,
            'bleu_2': self.bleu_2,
            'bleu_3': self.bleu_3,
            'bleu_4': self.bleu_4
        }

                


    def calc_perc_compiles(self, folder, fix_errors=False):
        code_folder = os.path.join('output', folder, 'code')
        compile_folder = os.path.join('output', folder, 'compiled')
        os.makedirs(compile_folder, exist_ok=True)

        subprocess.run('for entry in ' +  f'"{code_folder}"/*' + """
            do
            """ + f"{'#' if not fix_errors else ''} clang-tidy $entry -fix-errors" + """
            : ${entry/cpp/out} 
            g++ $entry -o  ${_/code/compiled}
            done""", shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return len(os.listdir(compile_folder)) / len(os.listdir(code_folder))



    def reconstructions_to_code(self, reconstructions, add_main=False):
        programs = []

        for tree in reconstructions:
            code = ''

            if add_main:
                self._add_main_to_reconstruction(tree)

            try:
                for child in tree.children:
                    code += self.parser.parse_node(child)
            except Exception as e:
                pass

            programs.append(code)

        return programs


    def reconstructions_to_file(self, reconstructions, folder, ids):
        code_folder = os.path.join('output', folder, 'code')
        os.makedirs(code_folder, exist_ok=True)


        exporter = JsonExporter(indent=2)
        os.makedirs(os.path.join('output', folder, 'trees'), exist_ok=True)

        for idx, recon in zip(ids, reconstructions):
            with open(os.path.join('output', folder, 'trees', f'{idx}.json'), 'w') as f:
                f.write(exporter.export(recon))

            

        imports = ['using namespace std;', '#include <vector>', '#include <iostream>', '#include <string>',
            '#include <cstring>', '#include <queue>', '#include <stdio.h>', '#include <math.h>', '#include <map>', '#include <set>', '#include <stack>']

        for idx, program in zip(ids, self.reconstructions_to_code(reconstructions, add_main=True)):
            program_path = os.path.join(code_folder, f'{idx}.cpp')

            with open(program_path, 'w') as f:
                f.write(program)


            add_includes_usings(program_path, imports)

        self.programs = []


    def _add_main_to_reconstruction(self, reconstruction):
        """
            Set the name of the first function declaration that has a body (compount statement)
            to the name "main"
        """
        func_decl_names = findall(reconstruction, filter_=lambda node: node.parent is not None 
                                   and node.parent.token == self.vocabulary.token2index['RES']['NAME'] 
                                   and node.parent.parent is not None 
                                   and node.parent.parent.token == self.vocabulary.token2index['RES']['FUNCTION_DECL']
                                   and node.parent.parent.children[-1].token == self.vocabulary.token2index['RES']['COMPOUND_STMT']
                                   and node.parent.parent.children[1].token == self.vocabulary.token2index['RES']['TYPE_KIND']
                                   and node.parent.parent.children[1].children[0].token == self.vocabulary.token2index['RES']['TYPE']
                                   and len(node.parent.parent.children[1].children[0].children) > 0
                                   and node.parent.parent.children[1].children[0].children[0].token == self.vocabulary.token2index['TYPE']['int']
                                   )

        for name in func_decl_names:
            if len(findall(reconstruction, filter_=lambda node: node.token == name.token
                                   and node.parent is not None 
                                   and node.parent.token == self.vocabulary.token2index['RES']['REF'])) == 0:
                      name.token = 'main'          
                      break   


        # if len(func_decl_names) > 0:
        #     func_decl_names[0].token = 'main'


    def add_eval_hypotheses(self, orig_programs, folder, ids):
        code_folder = os.path.join('output', folder, 'original code')
        os.makedirs(code_folder, exist_ok=True)

        for idx, program in zip(ids, orig_programs):
            filtered_program = self._filter_program(program, os.path.join(code_folder, f'{idx}.cpp'))

            tokens = self.tokenizer.tokenize(filtered_program)

            tokens_list = []

            for token in tokens:
                value = token.token_value
                if token.token_type == TokenType.IDENTIFIER and len(value) > 1 and value.startswith(';'):
                    tokens_list.append(';')
                    tokens_list.append(value[1:])

                elif token.token_type == TokenType.IDENTIFIER and len(value) > 1 and value.endswith(';'):
                    tokens_list.append(value[:-1])
                    tokens_list.append(';')

                else:
                    tokens_list.append(value)

            self.ids.append(idx)
            self.bleu_hypotheses.append(tokens_list)


    def add_eval_references(self, asts, declared_names):
        asts = deepcopy(asts)
        for program, decl_names in zip(asts, declared_names):
            self._plugin_original_names(program, decl_names)


        for program in self.reconstructions_to_code(asts):
            tokens = self.tokenizer.tokenize(program)

            tokens_list = []

            for token in tokens:
                value = token.token_value
                if token.token_type == TokenType.IDENTIFIER and len(value) > 1 and value.startswith(';'):
                    tokens_list.append(';')
                    tokens_list.append(value[1:])

                elif token.token_type == TokenType.IDENTIFIER and len(value) > 1 and value.endswith(';'):
                    tokens_list.append(value[:-1])
                    tokens_list.append(';')

                else:
                    tokens_list.append(value)

            self.bleu_references.append([tokens_list])


    def generations_to_file(self, generations, folder, plugin_names=False):
        generations_folder = os.path.join('output', folder, 'generations')
        os.makedirs(generations_folder, exist_ok=True)

        if plugin_names:
            for program in generations:
                self._plugin_generation_names(program)
            

        imports = ['using namespace std;', '#include <vector>', '#include <iostream>', '#include <string>',
            '#include <cstring>', '#include <queue>', '#include <stdio.h>', '#include <math.h>', '#include <map>', '#include <set>', '#include <stack>']


        for program in self.reconstructions_to_code(generations, add_main=True):
            idx = len(os.listdir(generations_folder))
            program_path = os.path.join(generations_folder, f'{idx}.cpp')

            with open(program_path, 'w') as f:
                f.write(program)


            add_includes_usings(program_path, imports)


    def _plugin_original_names(self, ast, declared_names):
        if 'NAME_' in str(ast.token):
            name = declared_names.get_name(int(ast.token.split('_')[-1]))

            if name != -1:
                ast.token = name

        for child in ast.children:
            self._plugin_original_names(child, declared_names)


    def _plugin_generation_names(self, ast):
        if 'NAME_' in str(ast.token):
            index = int(ast.token.split('_')[-1])



            if index != -1:
                ast.token = self.vocabulary.index2token['NAME'][index]
                # Temporary fix, if unwanted token, get token at 100 points later
                if ast.token in ['main']:
                    ast.token = self.vocabulary.index2token['NAME'][index + 100]

        for child in ast.children:
            self._plugin_generation_names(child)


    def _build_ast(self, adj_list, features, vocabs, index=0, parent_node=None):
        children = adj_list[adj_list[:, 0] == index][:, 1]

        token = features[index].item()

        if parent_node is not None and len(children) == 0:
            parent_token = self.vocabulary.index2token['RES'][parent_node.token]
            if not ('LITERAL' in parent_token or parent_token == 'TYPE' or parent_token == 'REF_BUILTIN'):
                token = f'NAME_{features[index].item()}'
            
        node = Node(token, is_reserved=vocabs[index] == 'RES', parent=parent_node)

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


    def _filter_program(self, program, temp_file_path):
        # Things that are either already defined in other files and hence will not preprocess correctly
        # Or are defined in other version of C (e.g. VS code extension) and needs to be transformed to format that g++ understands
        manual_defines = [('EOF', '(-1)'), ('__int64', 'long long'), ('%I64d', '%lld')]

        for k,v in manual_defines:
            program = program.replace(k, v)


        # Create a temporary file to store program in
        temp_file = open(temp_file_path, 'w')
        temp_file.write(program)
        temp_file.close()

        # Call preprocess g++ function to expand the macros (need this to get operator tokens)
        preprocessed_program = subprocess.check_output(['g++', '-x', 'c++', '-E', temp_file_path], shell=False).decode()

        # Only retrieve the actual original code from the program not all the includes
        program_lines = preprocessed_program.split('\n')

        preprocessed_program_filtered = ''

        # Select only the lines of the original program
        skip = False
        for l in program_lines:
            if not skip and not l.startswith("#") and not l.startswith("using") and len(l.strip()) > 0:
                preprocessed_program_filtered += f'{l}\n'
            if l.startswith("# "):
                toks = l.strip().split(" ")
                linenum, filename = toks[1:3]
                flags = toks[3:]
                skip = "3" in flags

        return preprocessed_program_filtered
        
        

