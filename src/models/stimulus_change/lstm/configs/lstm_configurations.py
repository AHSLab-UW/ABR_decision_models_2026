from itertools import product
import json

hidden_sizes = [64, 128, 256]
lstm_layers = [2, 3, 4]

pre_fc_options = [
    [],
    [64],
    [128, 64],
    [256, 128, 64],
]

post_fc_options = [
    [],
    [64],
    [128, 64],
    [32, 128, 32]
]

label_embedding_dim = [
    16, 32, 64
]

pos_embedding_dim = [
    16, 32, 64
]

cnn_embed_dim = [
    96, 128
]

dropouts = [0.1, 0.2, 0.4]
activations = ["relu"]
pool_k_list = [8, 12]
learning_rates = [0.001, 0.0005, 0.0001]
epochs = [25]
scheduled_sampling = ["linear"]
prev_preds = [True, False]

model_configs = []

num_classes = 14 # output number of classes

for hs, nl, pre, post, dp, emb_dim, act, pos_emb_dim, prev_pred, sampling, epoch, lr, cnn_dim, pool_k in product(
        hidden_sizes,
        lstm_layers,
        pre_fc_options,
        post_fc_options,
        dropouts,
        label_embedding_dim,
        activations,
        pos_embedding_dim,
        prev_preds,
        scheduled_sampling,
        epochs,
        learning_rates,
        cnn_embed_dim,
        pool_k_list,):

    cfg = {
        "input_size": 12,# + num_classes,
        "cnn_embed_dim": cnn_dim,
        "waveform_length": 256,
        "pool_k": pool_k,
        "output_size": num_classes,
        "pre_fc_layers": pre,
        "lstm_hidden_size": hs,
        "holdout_validation": False,
        "prev_preds": prev_pred,
        "lstm_layers": nl,
        "label_embedding_dim": emb_dim,
        "epochs": epoch,
        "learning_rate": lr,
        "pos_embedding_dim": pos_emb_dim,
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
        "sampling_strategy": sampling,
    }
    model_configs.append(cfg)
