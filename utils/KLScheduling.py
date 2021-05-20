import math

class ConstantAnnealing:
    """
        Constant annealing schedule:
        set β to a constant fixed value during training

        Args:
            kl_weight: β value
    """

    def __init__(self, kl_weight=1):
        self.kl_weight = kl_weight

    def get_weight(self, current_iteration):
        return self.kl_weight


class MonotonicAnnealing:
    """
        Monotonic annealing schedule:
        set β close to 0 in the early stage of training, then monotically increase β until it reaches 1.

        Args:
            total_iter: Total number of iterations used in training (independent of epochs)
            ratio: proportion of total_iter used to increase β to 1, after that it is 1 (default 0.2)
            function: Function used to increase β by, e.g. linear or sinusoidal (default linear)
    """

    def __init__(self, total_iter, nr_warmup_iterations=0, ratio=0.2, function='linear'):
        self.total_iter = total_iter
        self.ratio = ratio
        self.nr_warmup_iterations = nr_warmup_iterations

        if function == 'linear':
            self.function = lambda tau: tau / ratio
        elif function == 'sinusoidal':
            self.function = lambda tau: (1 + math.sin(((tau/ratio) * math.pi) - (math.pi / 2))) / 2
        else:
            raise ValueError('Invalid function, choose: "linear" or "sinusoidal"')


    def get_weight(self, current_iteration):
        if current_iteration > self.nr_warmup_iterations:
            tau = (current_iteration - self.nr_warmup_iterations) / self.total_iter

            if tau <= self.ratio:
                return self.function(tau)
            else:
                return 1
        else:
            return 0


class CyclicalAnnealing:
    """
        Cyclical annealing schedule: https://arxiv.org/abs/1903.10145

        We start with β = 0, increase β at a fast pace, and then stay at β = 1 for subsequent
        learning iterations. Repeat this in cycles.

        Args:
            total_iter: Total number of iterations used in training (independent of epochs)
            cycles: The number of cycles to anneal from 0 to 1 (default 4)
            ratio: proportion used to increase β within a cycle (default 0.5)
            function: Function used to increase β by, e.g. linear or sigmoid (default linear)
    """

    def __init__(self, total_iter, nr_warmup_iterations=0, cycles=4, ratio=0.5, function='linear'):
        self.total_iter = total_iter
        self.cycles = cycles
        self.ratio = ratio
        self.nr_warmup_iterations = nr_warmup_iterations

        if function == 'linear':
            self.function = lambda tau: tau / ratio
        elif function == 'sinusoidal':
            self.function = lambda tau: (1 + math.sin(((tau/ratio) * math.pi) - (math.pi / 2))) / 2
        else:
            raise ValueError('Invalid function, choose: "linear" or "sinusoidal"')

    def get_weight(self, current_iteration):
        if current_iteration > self.nr_warmup_iterations:
            iters_per_cycle = (self.total_iter - self.nr_warmup_iterations)/self.cycles
            tau = (current_iteration - self.nr_warmup_iterations) % math.ceil(iters_per_cycle)
            tau /= iters_per_cycle

            if tau <= self.ratio:
                return self.function(tau)
            else:
                return 1
        else:
            return 0
