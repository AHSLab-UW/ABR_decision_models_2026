from itertools import product

# ----------------------------
# Hyperparameters for Wave 5 Transformer
# Non-autoregressive, causal attention
# Binary classification: does this waveform have Wave V?
#
# Best known config: dim=64, layers=3, heads=4, ff=128, dropout=0.25, lr=0.0005
#   → ~102K params → 83.40% test accuracy
# dim=32 (50K) → 82.82% | dim=128 (533K) overfit → val peaked 80.91%
# Sweet spot is dim=64 — compact enough to generalise, enough capacity for 12 features.
# ----------------------------

model_configs = [
    {
        "input_size": 12,
        "output_size": 1,
        "pre_fc_layers": [],
        "transformer_dim": 64,
        "transformer_layers": 3,
        "holdout_validation": False,
        "pos_embedding_dim": 8,
        "transformer_heads": 4,   # 64/4 = 16 dims per head
        "transformer_ff": 128,
        "post_fc_layers": [],
        "dropout": 0.25,
        "activation": "relu",
    }
]

learning_rates = [0.0005]
epochs = [100]
