from sacred import Experiment
from sacred.observers import MongoObserver

EXPERIMENT_NAME = 'Experiment'
YOUR_CPU = None
DATABASE_NAME = 'Autoencoder_program_synthesis'

ex = Experiment('Experiment')

ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))

@ex.config
def get_config():
    """
        Configuration of experiment used by sacred
    """

    # Standard model parameters
    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-4   
    num_rnn_layers_enc = 1  # The number of RNN layers for the encoder (>1 gives stacked RNN)
    num_rnn_layers_dec = 1  # The number of RNN layers for the decoder (>1 gives stacked RNN)
    rnn_hidden_size = 200   # The hidden size of the RNN
    latent_dim = 100        # The latent vector size
    embedding_dim = 50      # The embedding dimension (if pretrained embedding is set, will automatically take that size)
    clip_grad_norm = 0      # clip the gradient norm, setting to 0 ignores this 
    clip_grad_val = 0       # clip the gradient value, setting to 0 ignores this
    save_every = 1000       # Save per X batches
    save_dir = 'checkpoints'

    # Torch data loader parameters
    num_workers = 8                 # How many CPU cores will be used by the dataloader

    # Data parameters
    max_tree_size = 750             # Only use trees of size (in terms of nodes) below max tree size 
    max_name_tokens = 300         # Only use the most frequent max name tokens
    reusable_name_tokens = 100      # The rest of the name tokens are indexed to reusable ID's

    # Advanced model parameters
    pretrained_emb = 'glove-wiki-gigaword-50'   # Pretrained embedding to use (https://github.com/RaRe-Technologies/gensim-data/blob/master/list.json)
    vae = True                                  # Turning this off will revert to standard AE architecture
    use_cell_output_lstm = False                # In case of LSTM RNN, use also the cell state concatenated with the hidden state as output
    indiv_embed_layers = False                  # Use individual embedding layers for each vocab/node type
    indiv_rnn_layers_vocab_types = False        # Use individual RNN layers for each vocab/node type
    weighted_loss = False                       # Use weighted loss 


    # KL scheduling parameters
    kl_scheduling = 'cyclical'                  # Type of kl scheduler to use: constant, monotonic, cyclical
    kl_warmup_iters = 100                       # Give the model a number of iterations of kl weight 0 first to warm up
    kl_weight = 1                               # For constant annealing: the KL weight
    kl_ratio = 0.2                              # Ratio of total data/data in cycle used to increase kl weight to 1
    kl_function = 'linear'                      # kl scheduler increase function: linear, sinusoidal
    kl_cycles = 4                               # For cyclical kl scheduler, how many cycles to anneal


    # Early stopping parameters
    check_early_stop_every = 500                # Update early stop loss every X batches
    early_stop_patience = 3                     # how many steps to wait before stopping when loss is not improving
    early_stop_min_delta = 0                    # minimum difference between new loss and old loss for new loss to be considered as an improvement


    # Load pretrained model
    pretrained_model = 'checkpoints/100latent_200hidden/VAE_epoch2_batch3999_26-05-2021_1302.tar'


    # Data path parameters
    tokens_paths = {
        'NAME': '../data/ast_trees_full_19-05-2021/name_tokens/',
        'RES': '../data/ast_trees_full_19-05-2021/reserved_tokens/',
        'TYPE': '../data/ast_trees_full_19-05-2021/type_tokens/',
        'LITERAL': '../data/ast_trees_full_19-05-2021/literal_tokens/',
        # 'NAME': '../data/test_dataset/name_tokens/',
        # 'RES': '../data/test_dataset/reserved_tokens/',
        # 'TYPE': '../data/test_dataset/type_tokens/',
        # 'LITERAL': '../data/test_dataset/literal_tokens/',
    }

    dataset_paths = {
        'TRAIN': '../data/ast_trees_full_19-05-2021/asts_train/',
        'VAL': '../data/ast_trees_full_19-05-2021/asts_val/',
        'TEST': '../data/ast_trees_full_19-05-2021/asts_test/'
        # 'TRAIN': '../data/test_dataset/asts_train/',
        # 'VAL': '../data/test_dataset/asts_val/',
        # 'TEST': '../data/test_dataset/asts_test/'  
    } 



""" 
To start sacredboard:
sacredboard -m Autoencoder_program_synthesis

To start training a model with custom config:
python3 train.py with "num_epochs=5"

If connection gets refused try the following commands in terminal:
sudo rm /var/lib/mongodb/mongod.lock
sudo service mongod start

then try again :)
"""
