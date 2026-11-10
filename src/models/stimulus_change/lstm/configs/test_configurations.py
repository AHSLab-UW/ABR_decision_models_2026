from itertools import product

hidden_sizes = [256, 320]
lstm_layers = [2, 3]

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
    16, 24, 32
]

label_embedding_dim = [16, 24, 32]

dropouts = [0.1, 0.15]

activations = ['relu']

learning_rates = [0.0005]

epochs = [25]
scheduled_sampling = ["linear"]

model_configs_test = []

prev_preds = [True]

num_classes = 14

for hs, nl, pre, post, dp, emb_dim, act, pos_emb_dim, prev_pred, epoch, rate, sampling in product(
        hidden_sizes,
        lstm_layers,
        pre_fc_options,
        post_fc_options,
        dropouts,
        label_embedding_dim,
        activations,
        pos_embedding_dim,
        prev_preds,
        epochs,
        learning_rates,
        scheduled_sampling):

    cfg = {
        "input_size": 12,# + num_classes,
        "output_size": num_classes,
        "pre_fc_layers": pre,
        "lstm_hidden_size": hs,
        "prev_preds": prev_pred,
        "holdout_validation": False,
        "lstm_layers": nl,
        "label_embedding_dim": emb_dim,
        "pos_embedding_dim": pos_emb_dim,
        "post_fc_layers": post,
        "dropout": dp,
        "epochs": epoch,
        "scheduled_sampling": sampling,
        "learning_rate": rate,
        "activation": act,
    }
    model_configs_test.append(cfg)