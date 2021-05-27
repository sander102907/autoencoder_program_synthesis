import torch.nn.functional as F
import torch

"""
    Sampling for decoding generation
    currently only works on batch size of 1, so single tokens

    Args:
        - Temperature:      how strongly we sample from the distribution (at high temperatures,
                            everything is uniform, at low temperatures below 1,
                            small differences are magnified)

        - Top-k:            Sample from only top-k tokens in probability distribution

        - Top-p :           Sample from a selection of the highest probability tokens whose
                            cumulative mass is larger than p (nucleus filtering)
"""

class Sampling:
    @classmethod
    def _get_sample(cls, logits):
        """

        """
        
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, 1)


    @classmethod
    def _filter_top_k(cls, logits, top_k):
        """
            Set logits below the top-k logits to -inf so softmax sets their probability to 0
        """

        # safety check
        top_k = min(top_k, logits.size(-1))

        if top_k > 0:
            # Get the indices of the logits below the top-k logits
            indices_below_top_k = logits < torch.topk(logits, top_k)[0][..., -1, None]

            # Set the logits below the top-k logits to -infinity so softmax gives them probability 0
            logits[indices_below_top_k] = float('-inf')

        return logits

    
    @classmethod
    def _filter_top_p(cls, logits, top_p):
        """
            Set logits above the top-p proability mass to -inf so softmax sets their probability to 0
        """

        if top_p > 0.0:
            # Sort the logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)

            # Run through softmax to get probabilities and then get the cumulative probabilities
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing and get the indices of the logits below the top-p logits
            indices_above_top_p = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

            # Set the logits below the top-p logits to -infinity so softmax gives them probability 0
            logits[indices_above_top_p] = float('-inf')
            
        return logits


    @classmethod
    def sample(cls, logits, temperature=1.0, top_k=0, top_p=0.0):
        """
            Get sample from logits with temperature control and nucleus filtering (top-k and top-p)
        """

        if logits.shape[0] > 0:
            logits /= temperature
            
            logits = cls._filter_top_k(logits, top_k)
            logits = cls._filter_top_p(logits, top_p)

            return cls._get_sample(logits)
        else:
            # Return empty long tensor if logits are empty
            return torch.empty(0, dtype=torch.long, device=logits.device)

        

               


