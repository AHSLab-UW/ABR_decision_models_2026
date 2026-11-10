import sys
import os
import argparse
import pandas as pd
import numpy as np

from src.common.model.stimulus_change_model_util import *
from src.models.stimulus_change.dnn.configurable_dnn_model import ConfigurableDNNModel
from src.models.stimulus_change.dnn.configs.dnn_configurations import *
from src.preprocess.pca_tuned import extract_features_train_test

if __name__ == "__main__":
    #TODO encapsulate this in our model config section
    parser = argparse.ArgumentParser(description="Train DNN with configurable dataset balancing.")
    parser.add_argument("--balance", type=str, choices=["true", "false"], default="false",
                        help="Toggle dataset balancing (undersample repeats, oversample minorities).")
    args = parser.parse_args()

    print(f"Initializing data pipeline...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

    action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/action_table.csv",
                            dtype=np.int32)
    input_data = pd.concat(load_dfs(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/full_table"), axis=0)
    #input_data = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/input.csv", dtype=np.float32)

    input_df, big_merged_df, label_ids = get_inputs_outputs(input_data, action_df)
    mapping, reverse_mapping = get_label_mappings(big_merged_df)
    labels_categorical = np.array(label_ids.map(mapping))
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]

    print("Splitting into Train/Test/Holdout and applying Global Scaler...")
    train_df, test_df, holdout_df = test_train_holdout_split_df(big_merged_df)
    print("Applying PCA to waveform columns...")
    waveform_cols = big_merged_df.columns[3:47].tolist()
    train_df, test_df, pca_model = extract_features_train_test(
        train_df, test_df, waveform_cols, n_components=10
    )
    _, holdout_df, _ = extract_features_train_test(
        None, holdout_df, waveform_cols, n_components=10, pca_model=pca_model
    )
    joblib.dump(pca_model, f'{ROOT_DIR}/exported_features/stimulus_change/preprocess/pca_model_dnn.joblib')

    print("Applying Global Scaler...")
    scaler = scale_dfs_2([train_df], [test_df], [holdout_df])
    joblib.dump(scaler, f'{ROOT_DIR}/exported_features/stimulus_change/preprocess/scaler_dnn.joblib')

    is_balanced = (args.balance.lower() == "true")

    if is_balanced:
        print("Applying dataset balancing...")
        repeats = len(list(filter(lambda x: x == 7, labels_categorical)))
        print(f"Amount of repeats before balance: {repeats}, Percent: {repeats / len(labels_categorical) * 100:.2f}%")

        train_df = undersample_class(train_df, 7, 0.20)
        train_df = oversample_minorities(train_df, 10000)
    else:
        print("Using standard, unbalanced dataset...")

    print("Extracting tensors and creating batches...")
    train_data, train_labels = split_df_data_labels(train_df, mapping)
    test_data, test_labels = split_df_data_labels(test_df, mapping)
    holdout_data, holdout_labels = split_df_data_labels(holdout_df, mapping)

    train_batches = create_batches(train_data, train_labels)
    test_batches = create_batches(test_data, test_labels)
    holdout_batches = create_batches(holdout_data, holdout_labels)

    print(f"Executing model tuning via grid search...")

    for config in model_configs:
        config["balanced_dataset"] = is_balanced

    evaluate_grid_search(
        model_configs=[best_config_dnn],
        model_factory=lambda cfg: ConfigurableDNNModel(cfg),
        train_split=train_batches,
        test_split=test_batches,
        holdout_split=holdout_batches,
        mapping=mapping,
        reverse_mapping=reverse_mapping,
        unique_labels=labels,
        labels_categorical=labels_categorical,
        root_dir=ROOT_DIR,
        namespace="dnn",
    )