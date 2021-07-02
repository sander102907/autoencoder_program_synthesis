from sacred import Experiment
from sacred.observers import MongoObserver

EXPERIMENT_NAME = 'Experiment'
DATABASE_NAME = 'Autoencoder_program_synthesis'
URL = f'mongodb+srv://sander:9AqrPVfuPJuv0ajP@cluster0.b2wvr.mongodb.net/{DATABASE_NAME}?retryWrites=true&w=majority'
# URL = None

ex = Experiment(EXPERIMENT_NAME)

ex.observers.append(MongoObserver.create(url=URL, db_name=DATABASE_NAME))

@ex.config
def get_config():
    """
        Configuration of experiment used by sacred
    """

    # Standard model parameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4   
    num_rnn_layers_enc = 3  # The number of RNN layers for the encoder (>1 gives stacked RNN)
    num_rnn_layers_dec = 3  # The number of RNN layers for the decoder (>1 gives stacked RNN)
    rnn_hidden_size = 300   # The hidden size of the RNN
    latent_dim = 150        # The latent vector size
    embedding_dim = 50      # The embedding dimension (if pretrained embedding is set, will automatically take that size)
    dropout = 0.2             # Dropout rate
    recurrent_dropout = 0.2   # Dropout rate for the RNN layers
    clip_grad_norm = 0      # clip the gradient norm, setting to 0 ignores this 
    clip_grad_val = 0       # clip the gradient value, setting to 0 ignores this
    save_every = 1000        # Save per X batches
    save_dir = 'checkpoints'

    # Torch data loader parameters
    num_workers = 8                 # How many CPU cores will be used by the dataloader

    # Data parameters
    max_tree_size = 750             # Only use trees of size (in terms of nodes) below max tree size 
    max_name_tokens = 0             # Only use the most frequent max name tokens
    reusable_name_tokens = 150      # The rest of the name tokens are indexed to reusable ID's

    # Sequence data parameters
    max_program_size = 750          # Cut off last part of programs > max program size and pad the other programs to max program size

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
    kl_cycles = 16                               # For cyclical kl scheduler, how many cycles to anneal


    # Sampling parameters
    temperature = 0.1
    top_k = 40
    top_p = 0.9


    # Early stopping parameters
    check_early_stop_every = 8000                # Update early stop loss every X batches
    early_stop_patience = 3                     # how many steps to wait before stopping when loss is not improving
    early_stop_min_delta = 0                    # minimum difference between new loss and old loss for new loss to be considered as an improvement


    # Load pretrained model
    pretrained_model = None #'checkpoints/cluster_latent150/7000.tar'


    # Data path parameters
    tokens_paths = {
        'NAME': '../data/ast_trees_full_19-06-2021/name_tokens/',
        'NAME_BUILTIN': '../data/ast_trees_full_19-06-2021/name_builtin_tokens/',
        'RES': '../data/ast_trees_full_19-06-2021/reserved_tokens/',
        'TYPE': '../data/ast_trees_full_19-06-2021/type_tokens/',
        'LITERAL': '../data/ast_trees_full_19-06-2021/literal_tokens/',


        # For the seq2seq model
        # 'ALL' : '../data/seq_data/token_counts/',
    }

    dataset_paths = {
        'TRAIN': '../data/ast_trees_full_19-06-2021/asts_train/',
        'VAL': '../data/ast_trees_full_19-06-2021/asts_val/',
        'TEST': '../data/ast_trees_full_19-06-2021/asts_test/',
        'TEST_PROGRAMS': '../data/ast_trees_full_19-06-2021/programs_test.csv'


        # For the seq2seq model
        # 'TRAIN': '../data/seq_data/programs_train/',
        # 'VAL': '../data/seq_data/programs_val/',
        # 'TEST': '../data/seq_data/programs_test/'
    } 



""" 
To start sacredboard:
sacredboard -m Autoencoder_program_synthesis

from mongodb atlas hosted database:
sacredboard -mu mongodb+srv://sander:9AqrPVfuPJuv0ajP@cluster0.b2wvr.mongodb.net/ Autoencoder_program_synthesis

To start training a model with custom config:
python3 train.py with "num_epochs=5"

If connection gets refused try the following commands in terminal:
sudo rm /var/lib/mongodb/mongod.lock
sudo service mongod start

then try again :)
"""
