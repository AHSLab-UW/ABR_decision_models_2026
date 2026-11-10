import sys
import os
import argparse
import pandas as pd
import numpy as np

from src.common.model.stimulus_change_model_util import *
from src.models.stimulus_change.lstm.configs.lstm_configurations import model_configs
from src.models.stimulus_change.lstm.configurable_lstm_no_prev_preds_model import ConfigurableNoPrevPredsLSTMModel
from src.models.stimulus_change.lstm.configurable_lstm_prev_preds_model import ConfigurablePrevPredsLSTMModel
from src.preprocess.pca_tuned import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--autoregressive", type=str, choices=["true", "false"], default=None)
    args = parser.parse_args()

    print(f"Initializing data pipeline...")
    action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/action_table.csv", dtype=np.int32)
    #input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/input.csv", dtype=np.float32)
    input_df = pd.concat(load_dfs(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/full_table"), axis=0)

    merged_dfs, big_merged_df, label_ids = get_seq_split_inputs_outputs(action_df, input_df)

    mapping, reverse_mapping = get_label_mappings(big_merged_df)
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]
    print(f"Label Mapping Calculated!: {labels}")
    labels_categorical = np.array(label_ids.map(mapping))

    print(f"Splitting into 80-10-10 split...")
    train_dfs, test_dfs, holdout_dfs = test_train_holdout_split_list(merged_dfs)

    print(f"Fitting PCA on train set...")
    waveform_cols = train_dfs[0].columns[3:47].tolist()
    train_flat = pd.concat(train_dfs, axis=0, ignore_index=False)
    _, _, pca_model = extract_features_train_test(train_flat, None, waveform_cols, n_components=10)
    joblib.dump(pca_model, f'{ROOT_DIR}/exported_features/stimulus_change/preprocess/pca_model_lstm.joblib')

    print(f"Applying PCA to all sequences...")
    train_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in train_dfs]
    test_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in test_dfs]
    holdout_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in holdout_dfs]

    print(f"Fitting global train scaler on dataset...")
    scaler = scale_dfs_2(train_dfs, test_dfs, holdout_dfs)
    joblib.dump(scaler, f'{ROOT_DIR}/exported_features/stimulus_change/preprocess/scaler_lstm.joblib')

    class_index = 13
    print(f"Splitting train and test sets into sequences and labels...")
    train_split = prepare_sequences(train_dfs, mapping, class_index)
    test_split = prepare_sequences(test_dfs, mapping, class_index)
    holdout_split = prepare_sequences(holdout_dfs, mapping, class_index)

    print(f"Executing model tuning via grid search...")
    configs_to_run = [
        best_config_lstm
        #holdout_val_config_lstm
    ]
    if args.epochs is not None:
        print(f"Overriding config epochs to: {args.epochs}")
        for config in configs_to_run:
            config["epochs"] = args.epochs
    if args.autoregressive is not None:
        is_autoregressive = (args.autoregressive.lower() == "true")
        print(f"Overriding config prev_preds to: {is_autoregressive}")
        for config in configs_to_run:
            config["prev_preds"] = is_autoregressive

    model_supplier = (lambda config: ConfigurablePrevPredsLSTMModel(config) if config['prev_preds'] else ConfigurableNoPrevPredsLSTMModel(config))

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
                         namespace="lstm")