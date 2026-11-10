from itertools import product

# ----------------------------
# Hyperparameters for Wave 5 CNN + Transformer
# Replaces PCA with a learnable 1D CNN on raw 256-point waveform
# Binary classification: does this waveform have Wave V?
#
# Data profile:
#   ~175K records, ~25K sequences, avg length ~7, max 38
#   258 input features per timestep (freq, dB, 256 raw waveform points)
#   61/39 class split — mild imbalance
#
# CNN architecture (fixed, wider intermediate channels):
#   Conv(1→64) → Conv(64→128) → Conv(128→cnn_output_dim) → AdaptiveAvgPool → flat
#   With cnn_output_dim=64:  ~66K CNN params
#   With cnn_output_dim=128: ~83K CNN params
#
# Total param estimates (CNN + projection + transformer):
#   ~130K–300K depending on config — comfortably above the PCA transformer
# ----------------------------

# CNN output dim: how many features the CNN compresses each waveform to
cnn_output_dims = [64, 128]

# Transformer grid
transformer_dims = [64, 128]
transformer_layers = [2, 3]
transformer_heads = [4]           # 64 and 128 both divisible by 4
transformer_ff = [128, 256]
pre_fc_options = [[], [64]]
post_fc_options = [[]]
pos_embedding_dim = [8]
dropouts = [0.15, 0.25]
activations = ["relu"]

learning_rates = [0.0005, 0.001]
epochs = [100]  # was 30 — model hadn't converged, loss still falling at cutoff

model_configs = []

for cnn_dim, t_dim, n_layers, n_heads, ff_size, pre, post, dp, pos_emb_dim, act in product(
        cnn_output_dims,
        transformer_dims,
        transformer_layers,
        transformer_heads,
        transformer_ff,
        pre_fc_options,
        post_fc_options,
        dropouts,
        pos_embedding_dim,
        activations):

    cfg = {
        "cnn_output_dim": cnn_dim,
        "output_size": 1,
        "pre_fc_layers": pre,
        "transformer_dim": t_dim,
        "transformer_layers": n_layers,
        "holdout_validation": False,
        "pos_embedding_dim": pos_emb_dim,
        "transformer_heads": n_heads,
        "transformer_ff": ff_size,
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
    }
    model_configs.append(cfg)

