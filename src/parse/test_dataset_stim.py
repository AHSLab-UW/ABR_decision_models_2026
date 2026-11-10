# allows us to test an independent dataset on weights trained from a model
import sys
import argparse

from src.common.model.stimulus_change_model_util import *
from src.models.stimulus_change.dnn.configurable_dnn_model import ConfigurableDNNModel
from src.models.stimulus_change.dnn.configurable_random_model import ConfigurableRandomModel
from src.models.stimulus_change.lstm.configurable_lstm_no_prev_preds_model import ConfigurableNoPrevPredsLSTMModel
from src.models.stimulus_change.transformer.configurable_transformer_modle_no_prev_preds import \
    ConfigurableNoPrevPredsTransformerModel
from src.preprocess.pca_tuned import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

def load_weights(weight_path, model_supplier, config):
    model = model_supplier(config)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_stimulus_models():
    return (load_weights("../../exported_features/stimulus_change/weights/lstm_weights.pth", ConfigurableNoPrevPredsLSTMModel, best_config_lstm),
            load_weights("../../exported_features/stimulus_change/weights/transformer_weights.pth", ConfigurableNoPrevPredsTransformerModel, best_config_transformer),
            load_weights("../../exported_features/stimulus_change/weights/dnn_weights.pth", ConfigurableDNNModel, best_config_dnn))

def get_dataset_features(namespace):
    action_table = pd.read_csv(f"../../data/feature_tables/{namespace}/action_table.csv")
    input_df = pd.concat(load_dfs(f"../../data/feature_tables/{namespace}/full_table"), axis=0)
    return action_table, input_df

def export_cross_set_results(namespace, accuracy, results_logger, freq_logger, reverse_mapping, labels, seq_logger=None):
    out_dir = f"{ROOT_DIR}/exported_features/stimulus_change/cross_set/{namespace}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results_logger.get_results(), f, indent=4)
    with open(f"{out_dir}/freq_results.json", "w") as f:
        json.dump(freq_logger.get_binned_results_dict(), f, indent=4)

    freq_logger.show_plot(title=f"Accuracy by Frequency ({namespace})",
                          save_path=f"{out_dir}/freq_accuracy.png")

    if seq_logger is not None:
        with open(f"{out_dir}/sequence_results.json", "w") as f:
            json.dump(seq_logger.get_binned_results_dict(), f, indent=4)

        seq_logger.show_plot(
            title=f"Accuracy vs Sequence Progression ({namespace})",
            save_path=f"{out_dir}/sequence_accuracy.png")

    pred_df = results_logger.get_true_labels_preds()
    pred_df.to_csv(f"{out_dir}/pred_spread.csv", index=False)

    with open(f"{out_dir}/reverse_mapping.json", "w") as f:
        json.dump(reverse_mapping, f, indent=4)

    cm = results_logger.get_cm()
    title_text = f'Confusion Matrix ({namespace.upper()} Model) - {accuracy:.2f}'
    plot_cm(cm, title_text, 'Predicted Labels', 'True Labels', labels,
            save_path=f"{out_dir}/{namespace}_cm.png",
            should_save=True, holdout_validation_cm=False)

def cross_test_models_sequential(test_dataset_namespace, models):
    action_table, input = get_dataset_features(test_dataset_namespace)
    action_table.drop(["session_id"], inplace=True, axis=1)
    merged_dfs, big_merged_df, label_ids = get_seq_split_inputs_outputs(action_table, input)

    mapping, reverse_mapping = get_label_mappings(big_merged_df, idx=47)
    print(f"Label Mapping Calculated!: {sorted(reverse_mapping.keys())}")
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]

    waveform_cols = merged_dfs[0].columns[3:47].tolist()

    for namespace, model in models:
        pca_model = joblib.load(f"../../exported_features/stimulus_change/preprocess/pca_model_{namespace}.joblib")
        transformed_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in merged_dfs]

        scaler = joblib.load(f"../../exported_features/stimulus_change/preprocess/scaler_{namespace}.joblib")
        for df in transformed_dfs:
            df.iloc[:, [2]] = scaler.transform(df.iloc[:, [2]])

        split = prepare_sequences(transformed_dfs, mapping, 13)
        results_logger, freq_logger, seq_logger = testing_loop_sequential(model, split_indiv_records(split), reverse_mapping)
        accuracy = results_logger.summarize_results()
        freq_logger.summarize_results()
        seq_logger.summarize_results()

        export_cross_set_results(namespace, accuracy, results_logger, freq_logger, reverse_mapping, labels,
                                 seq_logger=seq_logger)

def cross_test_models_independent(test_dataset_namespace, models):
    action_table, input = get_dataset_features(test_dataset_namespace)
    action_table.drop(['session_id'], inplace=True, axis=1)
    input_df, big_merged_df, label_ids = get_inputs_outputs(input, action_table)
    mapping, reverse_mapping = get_label_mappings(big_merged_df, idx=47)
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]

    waveform_cols = big_merged_df.columns[3:47].tolist()

    for namespace, model in models:
        pca_model = joblib.load(f"../../exported_features/stimulus_change/preprocess/pca_model_{namespace}.joblib")
        transformed_df = apply_pca(big_merged_df, waveform_cols, 10, pca_model)

        scaler = joblib.load(f"../../exported_features/stimulus_change/preprocess/scaler_{namespace}.joblib")
        transformed_df.iloc[:, 3:13] = scaler.transform(transformed_df.iloc[:, 3:13])

        test_data, test_labels = split_df_data_labels(transformed_df, mapping)
        test_batches = create_batches(test_data, test_labels)

        results_logger, freq_logger = testing_loop_independent(model, test_batches, reverse_mapping)
        accuracy = results_logger.summarize_results()
        freq_logger.summarize_results()

        export_cross_set_results(namespace, accuracy, results_logger, freq_logger, reverse_mapping, labels)

def cross_test_random(test_dataset_namespace, preprocess_namespace="lstm"):
    # Random model ignores feature values entirely; PCA is applied only so the
    # label columns land at class_index 13 for prepare_sequences. The freq column
    # (read by the frequency logger) is untouched, so no scaling is needed.
    action_table, input = get_dataset_features(test_dataset_namespace)
    action_table.drop(["session_id"], inplace=True, axis=1)
    merged_dfs, big_merged_df, label_ids = get_seq_split_inputs_outputs(action_table, input)

    mapping, reverse_mapping = get_label_mappings(big_merged_df, idx=47)
    print(f"Label Mapping Calculated!: {sorted(reverse_mapping.keys())}")
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]

    waveform_cols = merged_dfs[0].columns[3:47].tolist()
    pca_model = joblib.load(f"{ROOT_DIR}/exported_features/stimulus_change/preprocess/pca_model_{preprocess_namespace}.joblib")
    transformed_dfs = [apply_pca(df, waveform_cols, 10, pca_model) for df in merged_dfs]

    split = prepare_sequences(transformed_dfs, mapping, 13)

    random_model = ConfigurableRandomModel(best_config_random)
    random_model.eval()

    results_logger, freq_logger, seq_logger = testing_loop_sequential(random_model, split_indiv_records(split), reverse_mapping)
    accuracy = results_logger.summarize_results()
    freq_logger.summarize_results()
    seq_logger.summarize_results()

    export_cross_set_results("random", accuracy, results_logger, freq_logger, reverse_mapping, labels,
                             seq_logger=seq_logger)

if __name__ == "__main__":
    all_models = ["lstm", "transformer", "dnn", "random"]
    parser = argparse.ArgumentParser(description="Cross-test trained stimulus models on an independent dataset.")
    parser.add_argument("--dataset", type=str, default="uw", help="Feature table namespace to test on.")
    parser.add_argument("--models", type=str, default=",".join(all_models),
                        help=f"Comma-separated subset of {all_models}.")
    args = parser.parse_args()
    selected = [m.strip() for m in args.models.split(",")]

    if any(m in selected for m in ["lstm", "transformer", "dnn"]):
        lstm_model, transformer_model, dnn_model = get_stimulus_models()
        sequential_models = [(ns, m) for ns, m in [('lstm', lstm_model), ('transformer', transformer_model)] if ns in selected]
        if sequential_models:
            cross_test_models_sequential(args.dataset, sequential_models)
        if "dnn" in selected:
            cross_test_models_independent(args.dataset, [('dnn', dnn_model)])

    if "random" in selected:
        cross_test_random(args.dataset)
