import sys
import os
import argparse
import pandas as pd
import numpy as np

from src.common.model.stimulus_change_model_util import *
from src.models.stimulus_change.lstm.configs.lstm_configurations import model_configs
from src.utils.waveform_loader import load_waveform_data

# Import your CNN-LSTM Hybrid models
from src.models.stimulus_change.lstm.lstm_cnn_hybrid_autoreg import ConfigurablePrevPredsCNNLSTMModel
from src.models.stimulus_change.lstm.lstm_cnn_hybrid_nonautoreg import ConfigurableNoPrevPredsCNNLSTMModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM with configurable scaling and epochs.")
    parser.add_argument("--scaling", type=str, choices=["causal", "global_delta", "train_scale", "improper", "none"],
                        default="causal",
                        help="Scaling method to use: 'causal', 'global_delta', or 'none'.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs to train for")
    parser.add_argument("--autoregressive", type=str, choices=["true", "false"], default=None,
                        help="Toggle previous predictions on/off")
    args = parser.parse_args()

    print(f"Initializing data pipeline...")
    action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/action_table.csv", dtype=np.int32)
    input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/input.csv", dtype=np.float32)

    print(f"Loading waveform tables...")
    _, _, waveform_df = get_dataset("sch")

    merged_dfs, big_merged_df, label_ids = get_seq_split_inputs_outputs(action_df, input_df)

    mapping, reverse_mapping = get_label_mappings(big_merged_df)
    print(f"Label Mapping Calculated!: {sorted(reverse_mapping.keys())}")
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]
    labels_categorical = np.array(label_ids.map(mapping))

    if args.scaling == "causal":
        print(f"Applying Causal Running Normalization...")
        scale_dfs_causal(merged_dfs)
    elif args.scaling == "improper":
        print(f"Applying Improper Running Normalization...")
        scale_dfs(merged_dfs)
    elif args.scaling == "none":
        print(f"Skipping scaling (Raw Data)...")

    print(f"Splitting into 80-10-10 split...")
    train_dfs, test_dfs, holdout_dfs = test_train_holdout_split_list(merged_dfs)

    class_index = 13

    if args.scaling == "global_delta":
        print(f"Applying Global + Delta Scaling (fitted on train set)...")
        scale_dfs_lstm_final(train_dfs, test_dfs, holdout_dfs)
        class_index = 24
    elif args.scaling == "train_scale":
        print(f"Fitting global train scaler on dataset")
        scale_dfs_2(train_dfs, test_dfs, holdout_dfs)

    print(f"Splitting train and test sets into (wave, meta, y) tuples...")
    train_split = prepare_sequences_waveform(train_dfs, mapping, waveform_df, class_index)
    test_split = prepare_sequences_waveform(test_dfs, mapping, waveform_df, class_index)
    holdout_split = prepare_sequences_waveform(holdout_dfs, mapping, waveform_df, class_index)

    print(f"Executing model tuning via grid search...")

    configs_to_run = model_configs

    if args.epochs is not None:
        print(f"Overriding config epochs to: {args.epochs}")
        for config in configs_to_run:
            config["epochs"] = args.epochs

    if args.scaling == "global_delta":
        for config in configs_to_run:
            config["input_size"] = 23

    if args.autoregressive is not None:
        is_autoregressive = (args.autoregressive.lower() == "true")
        print(f"Overriding config prev_preds to: {is_autoregressive}")
        for config in configs_to_run:
            config["prev_preds"] = is_autoregressive

    model_supplier = (lambda config: ConfigurablePrevPredsCNNLSTMModel(config) if config[
        'prev_preds'] else ConfigurableNoPrevPredsCNNLSTMModel(config))

    evaluate_grid_search(model_configs=configs_to_run,
                         model_factory=model_supplier,
                         train_split=train_split,
                         test_split=test_split,
                         holdout_split=holdout_split,
                         mapping=mapping,
                         reverse_mapping=reverse_mapping,
                         unique_labels=labels,
                         labels_categorical=labels_categorical,
                         root_dir=ROOT_DIR,
                         namespace="lstm_cnn")