class ModelResults:
    def __init__(self):
        self.bleu_4 = 0
        self.bleu_3 = 0
        self.bleu_2 = 0
        self.bleu_1 = 0

    def from_dict(self, dict):
        self.bleu_4 = dict['bleu_4']
        self.bleu_3 = dict['bleu_3']
        self.bleu_2 = dict['bleu_2']
        self.bleu_1 = dict['bleu_1']

    def __repr__(self):
        return f'Bleu 4: {round(self.bleu_4, 5)}'