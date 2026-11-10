from itertools import product

# ----------------------------
# Hyperparameters to sweep
# ----------------------------

# Size of the Transformer hidden dimension (d_model)
transformer_dims = [64, 128, 256]

# Number of Transformer layers (stacked encoder layers)
transformer_layers = [2, 3, 4]

# Number of attention heads in each multi-head self-attention
transformer_heads = [2, 4, 8]

# Feedforward network hidden size inside Transformer layers
transformer_ff = [128, 256, 512]

# Pre-FC layers before Transformer block
pre_fc_options = [
    [],
    [64],
    [128, 64],
    [256, 128, 64],
]

# Post-FC layers after Transformer block
post_fc_options = [
    [],
    [64],
    [128, 64],
    [32, 128, 32]
]

label_embedding_dims = [
    16, 32, 64
]

# Dropout rates
dropouts = [0.1, 0.2, 0.4]

# Activation function inside pre/post FC and Transformer FFN
activations = ["relu"]

# Learning rates, epochs, and scheduled sampling options
learning_rates = [0.001, 0.0005, 0.0001]
epochs = [25]
scheduled_sampling = ["linear"]

# Output classes
num_classes = 14

pos_embedding_dim = [
    16, 32, 64
]

model_configs = []

for dim, n_layers, n_heads, ff_size, pre, post, dp, emb_dim, pos_emb_dim, act, epoch, learning_rate, ss_strat in product(
        transformer_dims,
        transformer_layers,
        transformer_heads,
        transformer_ff,
        pre_fc_options,
        post_fc_options,
        dropouts,
        label_embedding_dims,
        pos_embedding_dim,
        activations,
        epochs,
        learning_rates,
        scheduled_sampling):

    cfg = {
        "input_size": 12,       # input + previous one-hot
        "output_size": num_classes,
        "pre_fc_layers": pre,
        "label_embedding_dim": emb_dim,
        "transformer_dim": dim,                # d_model
        "transformer_layers": n_layers,
        "holdout_validation": False,
        "pos_embedding_dim": pos_emb_dim,
        "transformer_heads": n_heads,         # num attention heads
        "transformer_ff": ff_size,             # feedforward hidden dim
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
        "epoch": epoch,
        "sampling_strategy": ss_strat,
        "learning_rate": learning_rate
    }
    model_configs.append(cfg)