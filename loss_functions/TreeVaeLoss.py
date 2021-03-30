import torch.nn as nn
import torch.nn.functional as F


class TreeVaeLoss(nn.Module):
    """
    Loss function for Tree VAE models
    Computes reconstruction loss based on:
        - tree node label prediction
        - parent topology prediction
        - successor sibling topolocy prediction
        
    Computes Kullback–Leibler divergence (kl_loss)
    
    Total loss = kl_loss + reconstruction_loss
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, output, mu, log_var, vocab_size): 
        # Negative log likelihood loss (categorical cross entropy)
        label_prediction_loss = F.nll_loss(output['predicted_labels'], output['labels'].long())
        
        # Binary cross entropy loss for parent and sibling predictions (topology)
        parent_loss = nn.BCELoss()(output['predicted_is_parent'], output['is_parent'])
        sibling_loss = nn.BCELoss()(output['predicted_has_siblings'], output['has_siblings'])
        
        reconstruction_loss =  label_prediction_loss + parent_loss + sibling_loss
        
        kl_loss = 0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))
        
        total_loss = kl_loss + reconstruction_loss
        
        losses = {
            'total_loss': total_loss,
            'label_prediction_loss': label_prediction_loss.item(),
            'parent_loss': parent_loss.item(),
            'sibling_loss': sibling_loss.item(),
            'kl_loss': kl_loss.item()
        }
                        
        return losses
        