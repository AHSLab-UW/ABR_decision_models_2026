from itertools import product

transformer_dims = [64, 128]

transformer_layers = [42, 3, 4]

transformer_heads = [8, 4, 8]

transformer_ff = [128, 256]

pre_fc_options = [
    [],
    [32],
]

post_fc_options = [
    [64],
    [32],
    [128, 64],
]

pos_embedding_dim = [
    16, 24, 32]

label_embedding_dim = [16, 24, 32]

dropouts = [0.1, 0.15]

activations = ['relu']

learning_rates = [0.0005]

epochs = [25]
scheduled_sampling = ["linear"]

# Output classes
num_classes = 14

model_configs_test = []

for dim, n_layers, n_heads, ff_size, pre, post, dp, emb_dim, pos_emb_dim, act, epoch, learning_rate, ss_strat in product(
        transformer_dims,
        transformer_layers,
        transformer_heads,
        transformer_ff,
        pre_fc_options,
        post_fc_options,
        dropouts,
        label_embedding_dim,
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
        "holdout_validation": False,
        "transformer_dim": dim,                # d_model
        "transformer_layers": n_layers,
        "pos_embedding_dim": pos_emb_dim,
        "transformer_heads": n_heads,         # num attention heads
        "transformer_ff": ff_size,             # feedforward hidden dim
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "sampling_strategy": ss_strat
    }
    model_configs_test.append(cfg)