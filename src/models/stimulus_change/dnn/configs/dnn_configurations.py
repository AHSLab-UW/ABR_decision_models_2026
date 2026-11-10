from itertools import product

hidden_layer_configs = [
    [32, 32],
    [64, 32],
    [64, 64, 32],
    [128, 64, 32],
    [128, 128, 64, 32]
]

pre_fc_options = [
    [],
    [16],
    [32],
    [64]
]

post_fc_options = [
    [],
    [32],
    [64, 32]
]

dropouts = [0.1, 0.2]
activations = ["relu", "leaky_relu", "gelu"]
use_batch_norm = [True, False]

learning_rates = [0.001, 0.0005, 0.0001]
epochs = [60]

sampling_strategies = ['none']
num_classes = 14
input_size = 12

model_configs = []

for (hl, pre, post, dp, act, bn, lr, epoch, s_strat) in product(
        hidden_layer_configs,
        pre_fc_options,
        post_fc_options,
        dropouts,
        activations,
        use_batch_norm,
        learning_rates,
        epochs,
        sampling_strategies):

    cfg = {
        "input_size": input_size,
        "output_size": num_classes,
        "pre_fc_layers": pre,
        "hidden_layers": hl,
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
        "use_batch_norm": bn,
        "learning_rate": lr,
        "epochs": epoch,
        "sampling_strategy": s_strat,
        "holdout_validation": False,
    }
    model_configs.append(cfg)

print(f"Total configurations generated: {len(model_configs)}")

# Example of how to access the first config
# print(model_configs[0])