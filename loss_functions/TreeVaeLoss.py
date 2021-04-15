import torch
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
    
    def __init__(self, kl_weight):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, output, mu, log_var, vocab_size): 
        # Negative log likelihood loss (categorical cross entropy)
        label_prediction_loss = F.nll_loss(output['predicted_labels'], output['labels'].long())
        
        # Binary cross entropy loss for parent and sibling predictions (topology)
        parent_loss = nn.BCELoss()(output['predicted_is_parent'], output['is_parent'])
        sibling_loss = nn.BCELoss()(output['predicted_has_siblings'], output['has_siblings'])
        
        reconstruction_loss =  label_prediction_loss + parent_loss + sibling_loss
        
        kl_loss = 0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))
        
        total_loss = self.kl_weight * kl_loss + reconstruction_loss
        
        losses = {
            'total_loss': total_loss,
            'label_prediction_loss': label_prediction_loss.item(),
            'parent_loss': parent_loss.item(),
            'sibling_loss': sibling_loss.item(),
            'kl_loss': kl_loss.item()
        }
                        
        return losses
    
class TreeVaeLossComplete(nn.Module):
    """
    Loss function for Tree VAE models
    Computes reconstruction loss based on:
        - tree node label prediction
        - leaf node label prediction
        - parent topology prediction
        - successor sibling topolocy prediction
        
    Computes Kullback–Leibler divergence (kl_loss)
    
    Total loss = kl_loss + reconstruction_loss
    """
    
    def __init__(self, kl_weight):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, output, mu, log_var):
        label_losses = {}

        for k, loss in output.items():
            if 'predicted_labels' in k:
                label_losses[k.split('_')[0] + '_loss'] = F.nll_loss(output[k], output[k.replace('_predicted', '')].long())
                print(k, output[k].shape, output[k.replace('_predicted', '')].long().shape, output[k])


        # Negative log likelihood loss (categorical cross entropy)
        # label_prediction_loss = F.nll_loss(output['predicted_labels'], output['labels'].long())
        
        # leaf_label_prediction_loss = F.nll_loss(output['predicted_labels_leaf'], output['labels_leaf'].long())
        
        # Binary cross entropy loss for parent and sibling predictions (topology)
        parent_loss = nn.BCELoss()(output['predicted_is_parent'], output['is_parent'])
        sibling_loss = nn.BCELoss()(output['predicted_has_siblings'], output['has_siblings'])

        reconstruction_loss =  parent_loss + sibling_loss

        for k, loss in label_losses.items():
            if k != 'LITERAL_loss':
                reconstruction_loss += loss

        # print(label_losses.keys())
        
        kl_loss = 0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))
        
        total_loss = self.kl_weight * kl_loss + reconstruction_loss
        
        losses = {
            'total_loss': total_loss,
            'parent_loss': parent_loss.item(),
            'sibling_loss': sibling_loss.item(),
            'kl_loss': kl_loss.item()
        }

        for k, loss in label_losses.items():
            losses[k] = loss.item()

        return losses
        
