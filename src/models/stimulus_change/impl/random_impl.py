import sys
import os
import argparse
import pandas as pd
import numpy as np

from src.common.model.stimulus_change_model_util import *
from src.models.stimulus_change.dnn.configurable_random_model import ConfigurableRandomModel
from src.preprocess.pca_tuned import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate random baseline (no training).")
    args = parser.parse_args()

    print(f"Initializing data pipeline...")
    action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/action_table.csv", dtype=np.int32)
    input_df = pd.concat(load_dfs(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/full_table"), axis=0)

    merged_dfs, big_merged_df, label_ids = get_seq_split_inputs_outputs(action_df, input_df)

    mapping, reverse_mapping = get_label_mappings(big_merged_df)
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]
    print(f"Label Mapping Calculated!: {labels}")
    labels_categorical = np.array(label_ids.map(mapping))

    print(f"Splitting into 80-10-10 split (session-level)...")
    train_dfs, test_dfs, holdout_dfs = test_train_holdout_split_list(merged_dfs)

    # Random model ignores feature values entirely, but we still need
    # the same column layout as the real models so prepare_sequences
    # produces correctly shaped tensors.
    print(f"Fitting PCA on train set (for consistent column layout only)...")
    waveform_cols = train_dfs[0].columns[3:47].tolist()
    train_flat = pd.concat(train_dfs, axis=0, ignore_index=False)
    _, _, pca_model = extract_features_train_test(train_flat, None, waveform_cols, n_components=10)

    print(f"Applying PCA to all sequences...")
    train_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in train_dfs]
    test_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in test_dfs]
    holdout_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in holdout_dfs]

    print(f"Scaling dB column...")
    scale_dfs_2(train_dfs, test_dfs, holdout_dfs)

    class_index = 13
    print(f"Splitting train and test sets into sequences and labels...")
    # train_split isn't used since we skip training, but built for parity / future use
    train_split = prepare_sequences(train_dfs, mapping, class_index)
    test_split = prepare_sequences(test_dfs, mapping, class_index)
    holdout_split = prepare_sequences(holdout_dfs, mapping, class_index)

    def evaluate_and_export(split, split_name):
        # fresh model per split so each split's results are reproducible on their own
        print(f"Instantiating random baseline model (no training)...")
        random_model = ConfigurableRandomModel(best_config_random)
        random_model.eval()

        print(f"Evaluating random baseline on {split_name} split...")
        results_logger, freq_logger, sequence_logger = testing_loop_sequential(
            random_model,
            split_indiv_records(split),
            reverse_mapping
        )

        accuracy = results_logger.summarize_results()
        freq_logger.summarize_results()
        sequence_logger.summarize_results()

        out_dir = f"{ROOT_DIR}/exported_features/stimulus_change/results/{split_name}/random"
        os.makedirs(out_dir, exist_ok=True)

        with open(f"{out_dir}/random_results.json", "w") as f:
            json.dump(results_logger.get_results(), f, indent=4)
        with open(f"{out_dir}/freq_results.json", "w") as f:
            json.dump(freq_logger.get_binned_results_dict(), f, indent=4)
        freq_logger.show_plot(title=f"Accuracy by Frequency (random - {split_name})",
                              save_path=f"{out_dir}/freq_accuracy.png")

        with open(f"{out_dir}/sequence_results.json", "w") as f:
            json.dump(sequence_logger.get_binned_results_dict(), f, indent=4)
        sequence_logger.show_plot(
            title=f"Accuracy vs Sequence Progression (random - {split_name})",
            save_path=f"{out_dir}/sequence_accuracy.png")

        pred_df = results_logger.get_true_labels_preds()
        pred_df.to_csv(f"{out_dir}/pred_spread.csv", index=False)

        with open(f"{out_dir}/reverse_mapping.json", "w") as f:
            json.dump(reverse_mapping, f, indent=4)

        cm = results_logger.get_cm()
        title_text = f'Confusion Matrix (Random Baseline - {split_name}) - {accuracy:.2f}'
        plot_cm(cm, title_text, 'Predicted Labels', 'True Labels', labels,
                save_path=f"{out_dir}/random_cm.png",
                should_save=True, holdout_validation_cm=(split_name == "holdout"))

        return accuracy

    test_accuracy = evaluate_and_export(test_split, "train")
    holdout_accuracy = evaluate_and_export(holdout_split, "holdout")

    print(f"Random baseline — Test Accuracy: {test_accuracy:.2f}%, Holdout Accuracy: {holdout_accuracy:.2f}%")