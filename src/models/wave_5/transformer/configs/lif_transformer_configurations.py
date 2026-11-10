from itertools import product

# ----------------------------
# Hyperparameters for Wave 5 LIF-Transformer
# Same non-autoregressive causal-attention architecture as the standard Transformer,
# but the input is expanded from 12 → 22 features by appending 10 causal
# Latency-Intensity Function (LIF) features derived from the clinician's Wave V markers
# (amplitude_5, latency_5) stored in the bank table.
#
# LIF feature set (10 dims, all computed causally — only past steps visible):
#   latency_5_curr      : Wave V latency at current step (0 if unmarked)
#   amplitude_5_curr    : Wave V amplitude at current step (0 if unmarked)
#   has_marking_curr    : 1 if current step has a valid clinician marking, else 0
#   n_prev_marked       : # of previous steps in this session with valid markings
#   lif_slope           : slope  of linear regression  latency ~ dB  (0 if <2 prior points)
#   lif_intercept       : intercept of the same regression
#   lif_pred_latency    : predicted latency at current dB from fitted LIF line
#   lif_residual        : latency_5_curr − lif_pred_latency  (0 if unmarked at current step)
#   avg_prev_amplitude  : mean amplitude from all prior marked steps (0 if none)
#   amp_slope           : slope of linear regression  amplitude ~ dB  (0 if <2 prior points)
#
# Coverage note: amplitude_5 / latency_5 come from the bank table in the database.
# In the committed CSV subset (~17K records from abr_mining_2) ~1.7% of wave-5
# rows have explicit markings.  Against the full Seattle Children's database the
# coverage is much higher.  The model handles the sparse case gracefully:
# has_marking_curr acts as a gate and LIF regression features default to 0
# when there are fewer than 2 prior marked points.
#
# Starting point for search: same dim=64 sweet-spot as the plain Transformer
# (83.40% with 12 features) — just widen input projection layer.
# ----------------------------

model_configs = [
    {
        "input_size": 22,           # 12 original + 10 LIF features
        "output_size": 1,
        "pre_fc_layers": [],
        "transformer_dim": 64,
        "transformer_layers": 3,
        "holdout_validation": False,
        "pos_embedding_dim": 8,
        "transformer_heads": 4,
        "transformer_ff": 128,
        "post_fc_layers": [],
        "dropout": 0.25,
        "activation": "relu",
    },
    {
        "input_size": 22,
        "output_size": 1,
        "pre_fc_layers": [32],      # lightweight projection before transformer
        "transformer_dim": 64,
        "transformer_layers": 3,
        "holdout_validation": False,
        "pos_embedding_dim": 8,
        "transformer_heads": 4,
        "transformer_ff": 128,
        "post_fc_layers": [],
        "dropout": 0.25,
        "activation": "relu",
    },
    {
        "input_size": 22,
        "output_size": 1,
        "pre_fc_layers": [],
        "transformer_dim": 128,     # larger model to absorb extra LIF dims
        "transformer_layers": 3,
        "holdout_validation": False,
        "pos_embedding_dim": 8,
        "transformer_heads": 4,
        "transformer_ff": 256,
        "post_fc_layers": [],
        "dropout": 0.25,
        "activation": "relu",
    },
]

learning_rates = [0.0005]
epochs = [100]
