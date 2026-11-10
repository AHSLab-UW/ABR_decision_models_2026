from itertools import product

# ----------------------------
# Hyperparameters for Wave 5 Autoregressive Transformer
# Causal + autoregressive: feeds previous prediction back via class embedding
# Binary classification: does this waveform have Wave V?
#
# Data profile:
#   ~175K records, ~25K sequences, avg length ~7, max 38
#   12 input features (freq, dB, 10 PCA components)
#   61/39 class split — mild imbalance
#
# Design rationale:
#   - Autoregressive loop is O(T^2) per sequence — keep model small
#   - Label embedding dim kept small (binary task, only 3 classes incl start token)
#   - Scheduled sampling combats exposure bias during training
# ----------------------------

# Focused grid: ~32 configs × 2 LRs × 2 sampling = 128 total runs
transformer_dims = [32, 64]
transformer_layers = [2, 3]
transformer_heads = [2, 4]
transformer_ff = [64, 128]
pre_fc_options = [[], [32]]
post_fc_options = [[]]
pos_embedding_dim = [8]
label_embedding_dim = [8]
dropouts = [0.15, 0.25]
activations = ["relu"]

learning_rates = [0.0005, 0.001]
epochs = [30]
scheduled_sampling = ["linear", "constant"]

model_configs = []

for dim, n_layers, n_heads, ff_size, pre, post, dp, pos_emb_dim, lbl_emb_dim, act in product(
        transformer_dims,
        transformer_layers,
        transformer_heads,
        transformer_ff,
        pre_fc_options,
        post_fc_options,
        dropouts,
        pos_embedding_dim,
        label_embedding_dim,
        activations):

    if dim % n_heads != 0:
        continue

    cfg = {
        "input_size": 12,
        "output_size": 1,
        "pre_fc_layers": pre,
        "transformer_dim": dim,
        "transformer_layers": n_layers,
        "holdout_validation": False,
        "pos_embedding_dim": pos_emb_dim,
        "label_embedding_dim": lbl_emb_dim,
        "transformer_heads": n_heads,
        "transformer_ff": ff_size,
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
    }
    model_configs.append(cfg)

