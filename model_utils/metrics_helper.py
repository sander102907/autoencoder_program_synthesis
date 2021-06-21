class MetricsHelper:
    @staticmethod
    def log_to_sacred(train, current_iteration, total_loss, kl_loss, reconstruction_loss, individual_losses,
                      accuracies, kl_weight, vocabs, _run):

        if train:
            loss_type = 'train'
        else:
            loss_type = 'validation' 

        _run.log_scalar(f'Total loss {loss_type}', total_loss.item(), current_iteration)
        _run.log_scalar(f'Reconstruction loss {loss_type}', reconstruction_loss.item(), current_iteration)
        _run.log_scalar(f'KL weight {loss_type}', kl_weight, current_iteration)
        _run.log_scalar(f'KL loss {loss_type}', kl_loss.item(), current_iteration)
        _run.log_scalar(f'Parent loss {loss_type}', individual_losses['PARENT'], current_iteration)
        _run.log_scalar(f'Sibling loss {loss_type}', individual_losses['SIBLING'], current_iteration)
        _run.log_scalar(f'Is reserved loss {loss_type}', individual_losses['IS_RES'], current_iteration)
        _run.log_scalar(f'Reserved label loss {loss_type}', individual_losses['RES'], current_iteration)
        _run.log_scalar(f'Type label loss {loss_type}', individual_losses['TYPE'], current_iteration)
        _run.log_scalar(f'Literal label loss {loss_type}', individual_losses['LITERAL'], current_iteration)
        _run.log_scalar(f'Name builtin label loss {loss_type}', individual_losses['NAME_BUILTIN'], current_iteration)
        _run.log_scalar(f'Name label loss {loss_type}', individual_losses['NAME'], current_iteration)

        _run.log_scalar(f'Total loss / node {loss_type}', total_loss.item() / len(vocabs), current_iteration)
        _run.log_scalar(f'Reconstruction loss / node {loss_type}', reconstruction_loss.item() / len(vocabs), current_iteration)
        _run.log_scalar(f'KL loss  / node {loss_type}', kl_loss.item() / len(vocabs), current_iteration)
        _run.log_scalar(f'Parent loss  / node {loss_type}', individual_losses['PARENT'] / len(vocabs), current_iteration)
        _run.log_scalar(f'Sibling loss  / node {loss_type}', individual_losses['SIBLING'] / len(vocabs), current_iteration)
        _run.log_scalar(f'Is reserved loss  / node {loss_type}', individual_losses['IS_RES'] / len(vocabs), current_iteration)
        _run.log_scalar(f'Reserved label loss  / node {loss_type}', individual_losses['RES'] / sum(vocabs == 'RES'), current_iteration)
        _run.log_scalar(f'Type label loss  / node {loss_type}', individual_losses['TYPE']  / sum(vocabs == 'TYPE'), current_iteration)
        _run.log_scalar(f'Literal label loss  / node {loss_type}', individual_losses['LITERAL']  / sum(vocabs == 'LITERAL'), current_iteration)
        _run.log_scalar(f'Name builtin label loss  / node {loss_type}', individual_losses['NAME_BUILTIN']  / sum(vocabs == 'NAME_BUILTIN'), current_iteration)
        _run.log_scalar(f'Name label loss  / node {loss_type}', individual_losses['NAME']  / sum(vocabs == 'NAME'), current_iteration)

        _run.log_scalar(f'Parent accuracy {loss_type}', accuracies['PARENT'], current_iteration)
        _run.log_scalar(f'Sibling accuracy {loss_type}', accuracies['SIBLING'], current_iteration)
        _run.log_scalar(f'Is reserved accuracy {loss_type}', accuracies['IS_RES'], current_iteration)
        _run.log_scalar(f'Reserved label accuracy {loss_type}', accuracies['RES'], current_iteration)
        _run.log_scalar(f'Type label accuracy {loss_type}', accuracies['TYPE'], current_iteration)
        _run.log_scalar(f'Literal label accuracy {loss_type}', accuracies['LITERAL'], current_iteration)
        _run.log_scalar(f'Name builtin label accuracy {loss_type}', accuracies['NAME_BUILTIN'], current_iteration) 
        _run.log_scalar(f'Name label accuracy {loss_type}', accuracies['NAME'], current_iteration)  


    @staticmethod
    def update_model_metrics(train, current_iteration, model_metrics, total_loss, kl_loss, reconstruction_loss, individual_losses,
                      accuracies, kl_weight, vocabs):

        if train:
            loss_type = 'train'
        else:
            loss_type = 'validation' 

        model_metrics[f'Total loss {loss_type}'][current_iteration] = total_loss.item()
        model_metrics[f'Reconstruction loss {loss_type}'][current_iteration] = reconstruction_loss.item()
        model_metrics[f'KL weight {loss_type}'][current_iteration] = kl_weight
        model_metrics[f'KL loss {loss_type}'][current_iteration] = kl_loss.item()
        model_metrics[f'Parent loss {loss_type}'][current_iteration] = individual_losses['PARENT']
        model_metrics[f'Sibling loss {loss_type}'][current_iteration] = individual_losses['SIBLING']
        model_metrics[f'Is reserved loss {loss_type}'][current_iteration] = individual_losses['IS_RES']
        model_metrics[f'Reserved label loss {loss_type}'][current_iteration] = individual_losses['RES']
        model_metrics[f'Type label loss {loss_type}'][current_iteration] = individual_losses['TYPE']
        model_metrics[f'Literal label loss {loss_type}'][current_iteration] = individual_losses['LITERAL']
        model_metrics[f'Name builtin label loss {loss_type}'][current_iteration] = individual_losses['NAME_BUILTIN']
        model_metrics[f'Name label loss {loss_type}'][current_iteration] = individual_losses['NAME']

        model_metrics[f'Total loss / node {loss_type}'][current_iteration] = total_loss.item() / len(vocabs)
        model_metrics[f'Reconstruction loss / node {loss_type}'][current_iteration] = reconstruction_loss.item() / len(vocabs)
        model_metrics[f'KL loss  / node {loss_type}'][current_iteration] = kl_loss.item() / len(vocabs)
        model_metrics[f'Parent loss  / node {loss_type}'][current_iteration] = individual_losses['PARENT'] / len(vocabs)
        model_metrics[f'Sibling loss  / node {loss_type}'][current_iteration] = individual_losses['SIBLING'] / len(vocabs)
        model_metrics[f'Is reserved loss  / node {loss_type}'][current_iteration] = individual_losses['IS_RES'] / len(vocabs)
        model_metrics[f'Reserved label loss  / node {loss_type}'][current_iteration] = individual_losses['RES'] / sum(vocabs == 'RES')
        model_metrics[f'Type label loss  / node {loss_type}'][current_iteration] = individual_losses['TYPE']  / sum(vocabs == 'TYPE')
        model_metrics[f'Literal label loss  / node {loss_type}'][current_iteration] = individual_losses['LITERAL']  / sum(vocabs == 'LITERAL')
        model_metrics[f'Name builtin label loss  / node {loss_type}'][current_iteration] = individual_losses['NAME_BUILTIN']  / sum(vocabs == 'NAME_BUILTIN')
        model_metrics[f'Name label loss  / node {loss_type}'][current_iteration] = individual_losses['NAME']  / sum(vocabs == 'NAME')

        model_metrics[f'Parent accuracy {loss_type}'][current_iteration] = accuracies['PARENT']
        model_metrics[f'Sibling accuracy {loss_type}'][current_iteration] = accuracies['SIBLING']
        model_metrics[f'Is reserved accuracy {loss_type}'][current_iteration] = accuracies['IS_RES']
        model_metrics[f'Reserved label accuracy {loss_type}'][current_iteration] = accuracies['RES']
        model_metrics[f'Type label accuracy {loss_type}'][current_iteration] = accuracies['TYPE']
        model_metrics[f'Literal label accuracy {loss_type}'][current_iteration] = accuracies['LITERAL']
        model_metrics[f'Name builtin label accuracy {loss_type}'][current_iteration] = accuracies['NAME_BUILTIN']
        model_metrics[f'Name label accuracy {loss_type}'][current_iteration] = accuracies['NAME']

    
    @staticmethod
    def init_model_metrics(model_metrics):
        for loss_type in ['train', 'validation']:
            model_metrics[f'Total loss {loss_type}'] = {}
            model_metrics[f'Reconstruction loss {loss_type}'] = {}
            model_metrics[f'KL weight {loss_type}'] = {}
            model_metrics[f'KL loss {loss_type}'] = {}
            model_metrics[f'Parent loss {loss_type}'] = {}
            model_metrics[f'Sibling loss {loss_type}'] = {}
            model_metrics[f'Is reserved loss {loss_type}'] = {}
            model_metrics[f'Reserved label loss {loss_type}'] = {}
            model_metrics[f'Type label loss {loss_type}'] = {}
            model_metrics[f'Literal label loss {loss_type}'] = {}
            model_metrics[f'Name builtin label loss {loss_type}'] = {}
            model_metrics[f'Name label loss {loss_type}'] = {}

            model_metrics[f'Total loss / node {loss_type}'] = {}
            model_metrics[f'Reconstruction loss / node {loss_type}'] = {}
            model_metrics[f'KL loss  / node {loss_type}'] = {}
            model_metrics[f'Parent loss  / node {loss_type}'] = {}
            model_metrics[f'Sibling loss  / node {loss_type}'] = {}
            model_metrics[f'Is reserved loss  / node {loss_type}'] = {}
            model_metrics[f'Reserved label loss  / node {loss_type}'] = {}
            model_metrics[f'Type label loss  / node {loss_type}'] = {}
            model_metrics[f'Literal label loss  / node {loss_type}'] = {}
            model_metrics[f'Name builtin label loss  / node {loss_type}'] = {}
            model_metrics[f'Name label loss  / node {loss_type}'] = {}

            model_metrics[f'Parent accuracy {loss_type}'] = {}
            model_metrics[f'Sibling accuracy {loss_type}'] = {}
            model_metrics[f'Is reserved accuracy {loss_type}'] = {}
            model_metrics[f'Reserved label accuracy {loss_type}'] = {}
            model_metrics[f'Type label accuracy {loss_type}'] = {}
            model_metrics[f'Literal label accuracy {loss_type}'] = {}
            model_metrics[f'Name builtin label accuracy {loss_type}'] = {}
            model_metrics[f'Name label accuracy {loss_type}'] = {}
