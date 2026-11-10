import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

import pandas as pd
import torch
import numpy as np
import json
import random

from src.common.model.wave_5_model_util import (
    prepare_sequences_wave5_cnn, scale_dfs_cnn,
    run_model, is_higher_than_saved_acc
)
from src.common.model.model_util_common import (
    test_train_holdout_split_list, split_action_table, print_model_architecture
)
from src.common.plot_util import plot_cm
from src.models.wave_5.transformer.cnn_transformer_model import Wave5CNNTransformerModel
from src.models.wave_5.transformer.configs.cnn_transformer_configurations import *

# ── Data Pipeline ──
# Instead of PCA features from input.csv, we load the raw 256-point waveforms
# from waveform.csv, join with bank → record_id, and add freq + dB from input.csv.

print("Initializing CNN-Transformer data pipeline...")
print("Loading waveform data (175K records × 256 points)...")

# Load raw waveform (avoid data_util's load_resource which forces dtype={9:str})
waveform_raw = pd.read_csv(f"{ROOT_DIR}/data/db_tables/seattle_childrens/waveform.csv")
banks_raw = pd.read_csv(f"{ROOT_DIR}/data/db_tables/seattle_childrens/bank.csv", dtype={9: str})

# Load action table (labels) and input.csv (for freq + dB metadata)
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/wave_5/wave_5_action_table.csv")
input_data = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/input.csv")

# Join waveform → bank to get record_id
waveform_cols = [c for c in waveform_raw.columns if c.startswith('value_')]
wf = pd.merge(
    waveform_raw[['bank_id'] + waveform_cols],
    banks_raw[['id', 'record_id']],
    left_on='bank_id', right_on='id'
)[['record_id'] + waveform_cols]

# Drop duplicate record_ids (if multiple banks per record, keep first)
wf = wf.drop_duplicates(subset='record_id', keep='first')

# Add freq and last_level from input.csv (freq is already log2-transformed)
meta = input_data[['record_id', 'freq', 'last_level']].drop_duplicates(subset='record_id')
cnn_input = pd.merge(wf, meta, on='record_id')

# Reorder: record_id, freq, last_level, value_0..value_255
cnn_input = cnn_input[['record_id', 'freq', 'last_level'] + waveform_cols]

# Fill any NaN waveform values with 0
cnn_input[waveform_cols] = cnn_input[waveform_cols].fillna(0.0)

print(f"CNN input table built: {cnn_input.shape[0]} records × {cnn_input.shape[1]} columns")

# Split into per-session sequences using action_df's stop column
split_sessions = split_action_table(action_df)
merged_dfs = [pd.merge(cnn_input, s, on='record_id') for s in split_sessions]
merged_dfs = [df for df in merged_dfs if len(df) > 0]

# Get labels for pos_weight calculation
big_merged = pd.merge(cnn_input, action_df, on='record_id')
label_ids = big_merged['has_wave_5']

mapping = {True: 1, False: 0}
reverse_mapping = {1: True, 0: False}
labels_categorical = np.array(label_ids.map(mapping))
labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]

# Scale: dB + waveform values (per session)
scale_dfs_cnn(merged_dfs)
print(f"Data scaled, splitting into 80-10-10 split ({len(merged_dfs)} sessions)")

train_dfs, test_dfs, val_dfs = test_train_holdout_split_list(merged_dfs)

print(f"Preparing sequence tensors...")
train_split = prepare_sequences_wave5_cnn(train_dfs, mapping)
test_split = prepare_sequences_wave5_cnn(test_dfs, mapping)
val_split = prepare_sequences_wave5_cnn(val_dfs, mapping)

# Free large DataFrames from memory
del waveform_raw, banks_raw, wf, cnn_input, big_merged, merged_dfs

print(f"Executing CNN-Transformer grid search...")


def evaluate_grid_search(model_configs, epochs, learning_rates):
    best_acc = 0
    plt_best = None
    i = 1
    total = len(model_configs) * len(epochs) * len(learning_rates)

    random.shuffle(model_configs)
    for config in model_configs:
        for lr in learning_rates:
            for ep in epochs:
                print(f"{i} / {total}: Epoch: {ep}, LR: {lr}, CNN dim: {config['cnn_output_dim']}")
                i += 1
                model = Wave5CNNTransformerModel(config)
                print_model_architecture(model, config)
                results_logger, sequence_logger = run_model(
                    model,
                    epochs=ep,
                    lr=lr,
                    train_set=train_split,
                    test_set=test_split,
                    label_set=labels_categorical,
                    holdout_validation=config['holdout_validation'],
                    sequential=True,
                    val_set=val_split
                )
                if config['holdout_validation']:
                    torch.save(model.state_dict(),
                               f'{ROOT_DIR}/exported_features/wave_5/weights/cnn_transformer_weights.pth')
                    for j in range(len(val_dfs)):
                        val_dfs[j].to_csv(
                            f"{ROOT_DIR}/data/cross_validation_sets/wave_5/cnn_transformer/test_df_{j}.csv",
                            index=False)
                    continue
                accuracy = results_logger.summarize_results()
                save_path = f"{ROOT_DIR}/exported_features/wave_5/results/train/cnn_transformer/cnn_transformer_results.json"
                if accuracy > best_acc and is_higher_than_saved_acc(accuracy, save_path):
                    print(f"Found better accuracy: {accuracy:.2f}%!")
                    sequence_logger.summarize_results()
                    results = results_logger.get_results()
                    cm = results_logger.get_cm()
                    torch.save(model.state_dict(),
                               f'{ROOT_DIR}/exported_features/wave_5/weights/cnn_transformer_weights.pth')
                    with open(f"{ROOT_DIR}/exported_features/wave_5/config/cnn_transformer_config.json", "w") as f:
                        json.dump(config, f, indent=4)
                    with open(save_path, "w") as f:
                        json.dump(results, f, indent=4)
                    with open(f"{ROOT_DIR}/exported_features/wave_5/results/train/cnn_transformer/sequence_results.json",
                              "w") as f:
                        json.dump(sequence_logger.get_binned_results_dict(), f, indent=4)

                    best_acc = accuracy
                    plt_best = cm
                    plot_cm(cm,
                            f'Confusion Matrix (Wave 5 CNN-Transformer) - {accuracy:.2f}',
                            'Predicted', 'True', labels,
                            save_path=f"{ROOT_DIR}/exported_features/wave_5/results/train/cnn_transformer/cnn_transformer_cm.png",
                            should_save=True, holdout_validation_cm=False)
                    sequence_logger.show_plot(f"{ROOT_DIR}/exported_features/wave_5/results/train/cnn_transformer/sequence_accuracy.png")
                    print("Found better accuracy, running cross-validation training to update weights")
                    config['holdout_validation'] = True
                    evaluate_grid_search([config], epochs, learning_rates)

    if plt_best is not None:
        print(best_acc)
        plot_cm(plt_best,
                f'Confusion Matrix (Wave 5 CNN-Transformer) - {best_acc:.2f}',
                'Predicted Labels', 'True Labels', labels,
                save_path=f"{ROOT_DIR}/exported_features/wave_5/results/train/cnn_transformer/cnn_transformer_cm.png",
                should_save=False, holdout_validation_cm=False)


evaluate_grid_search(model_configs, epochs, learning_rates)

