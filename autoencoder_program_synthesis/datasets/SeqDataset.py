from torch.utils.data import Dataset
import linecache
import subprocess
import json
import torch

class SeqDataset(Dataset):
    """Dataset for sequence models, loads source code as a sequence of tokens"""

    def __init__(self, json_file, vocabulary, max_program_size, device):
        """
            Args:
                json_file (string): Path to the json file with the programs
        """

        self._json_file = json_file
        self._total_data = int(subprocess.check_output(["jq", "length", json_file], shell=False)) - 1
        self.vocabulary = vocabulary
        self.max_program_size = max_program_size
        self.device = device

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):
        jsonstr = linecache.getline(self._json_file, idx + 2)
        linecache.clearcache()
        program = json.loads("{" + jsonstr[:-2] + "}")

        try:
            inp, target, length = self.__create_data(program)
        except Exception as e:
            print(e, idx, self._json_file)


        return {
            'input': inp,
            'target': target,
            'length': length,
            'id': list(program.keys())[0],
        }


    def __create_data(self, program):
        inp = ['<sos>'] + list(program.values())[0]
        inp = inp[:self.max_program_size]

        target = list(program.values())[0]
        target = target[:self.max_program_size-1] + ['<eos>']

        length = len(inp)

        inp.extend(['<pad>'] * (self.max_program_size - length))
        target.extend(['<pad>'] * (self.max_program_size - length))

        inp = torch.tensor([self.vocabulary.token2index['ALL'][t] if t in self.vocabulary.token2index['ALL'] else self.vocabulary.token2index['ALL']['<unk>'] for t in inp])
        target = torch.tensor([self.vocabulary.token2index['ALL'][t] if t in self.vocabulary.token2index['ALL'] else self.vocabulary.token2index['ALL']['<unk>']  for t in target])

        return inp, target, length




