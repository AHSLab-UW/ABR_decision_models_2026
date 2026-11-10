import copy

import numpy as np
from torch import nn, optim

from src.common.model.model_util_common import *
from src.common.model.results.aggregator.freq_results import FrequencyResults
from src.common.model.results.aggregator.sequence_results import SequenceResults
from src.common.model.results.logger.wave_5_results_logger import Wave5ClassificationResultsLogger
from src.common.plot_util import plot_cm, create_freq_graph


def get_sampling_probability(strategy, epoch, epochs):
    """Compute sampling probability for scheduled sampling (autoregressive models)."""
    if strategy == "linear":
        return min(0.5, epoch / epochs)
    if strategy == "constant":
        return 0.3
    if strategy == "none" or strategy is None:
        return 0
    return 0

def get_inputs_outputs(input, action_df):
    big_merged_df = pd.merge(left=input, right=action_df, on="record_id", how='inner')
    label_ids = big_merged_df.iloc[:, 13]
    create_freq_graph(label_ids)
    input_data = big_merged_df.iloc[:, :13]
    print(input_data.shape)
    return input_data, big_merged_df, label_ids

def get_seq_split_inputs_outputs(action_df, input):
    """Split data into per-session sequences using the 'stop' column, then merge with input features."""
    split_sessions = split_action_table(action_df)
    big_merged_df = pd.merge(left=input, right=action_df, on="record_id", how='left')
    label_ids = big_merged_df.iloc[:, 13]
    create_freq_graph(label_ids)
    merged_dfs = [pd.merge(input, df, on='record_id') for df in split_sessions]
    merged_dfs = list(filter(lambda x: len(x) > 0, merged_dfs))
    return merged_dfs, big_merged_df, label_ids

def prepare_sequences_wave5(dfs, label_map):
    """Convert per-session DataFrames into (features_tensor, binary_labels_tensor) tuples."""
    splits = []
    for df in dfs:
        df_labels = df.iloc[:, 13].map(label_map).to_numpy()
        x = torch.tensor(df.iloc[:, 1:13].values, dtype=torch.float32)
        y = torch.tensor(df_labels, dtype=torch.int64)
        splits.append((x, y))
    return splits

def split_df_data_labels(df, label_map):
    df_labels = df.iloc[:, 13]
    df_labels = df_labels.map(label_map).to_numpy()
    x = torch.tensor(df.iloc[:, 1:13].values, dtype=torch.float32)
    y = torch.tensor(df_labels, dtype=torch.int64)
    return x, y


def prepare_sequences_wave5_cnn(dfs, label_map):
    """Convert per-session DataFrames (with raw waveform columns) into (features_tensor, labels_tensor) tuples.

    Expects DataFrames with columns: record_id, freq, last_level, value_0..value_255, has_wave_5, stop.
    Features are [freq, last_level, value_0..value_255] = 258 dims.
    """
    waveform_cols = [f'value_{i}' for i in range(256)]
    feature_cols = ['freq', 'last_level'] + waveform_cols

    splits = []
    for df in dfs:
        labels = df['has_wave_5'].map(label_map).to_numpy()
        x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.int64)
        splits.append((x, y))
    return splits


def scale_dfs_cnn(merged_dfs):
    """Scale CNN metadata per-session: only freq (col 1) and dB (col 2).

    Waveform values (cols 3:259) are intentionally NOT scaled here — fitting a
    StandardScaler on ~7 rows × 256 columns produces garbage statistics.
    The CNN's BatchNorm layers handle waveform normalization internally.
    """
    for df in merged_dfs:
        df.iloc[:, [1]] = StandardScaler().fit_transform(df.iloc[:, [1]])
        df.iloc[:, [2]] = StandardScaler().fit_transform(df.iloc[:, [2]])


def run_model(model, epochs, lr, train_set, test_set, label_set, holdout_validation,
              sequential=False, val_set=None, sampling_strategy=None):
    """
    Train and evaluate a model.

    Args:
        sequential:        If True, uses the sequential training loop (iterates over full sessions)
                           and evaluates all positions in each sequence. Used for transformer/sequence models.
                           If False, uses the batched independent training loop. Used for DNN.
        val_set:           Optional validation set. If provided, val accuracy is computed every epoch,
                           best model is saved by val acc, and early stopping is based on val acc.
        sampling_strategy: Scheduled sampling strategy for autoregressive models ('linear', 'constant', or None).
    """
    if holdout_validation:
        print("Using holdout validation training, merging test/train sets")
        train_set = train_set + test_set
        test_set = val_set
        val_set = None
    if sequential:
        training_loop_sequential(epochs=epochs, model=model, lr=lr, train_set=train_set,
                                 label_set=label_set, val_set=val_set, sampling_strategy=sampling_strategy)
        return testing_loop_sequential(model, split_indiv_records(test_set))
    else:
        training_loop(epochs=epochs, model=model, lr=lr, train_set=train_set, label_set=label_set, val_set=val_set)
        return testing_loop_independent(model, test_set)


def _get_pos_weight(label_set):
    if isinstance(label_set, torch.Tensor):
        labels_np = label_set.cpu().numpy()
    else:
        labels_np = np.array(label_set)

    num_positives = np.sum(labels_np == 1)
    num_negatives = np.sum(labels_np == 0)
    pos_weight_val = num_negatives / num_positives if num_positives > 0 else 1.0
    return torch.tensor([pos_weight_val], dtype=torch.float32)


def training_loop(epochs, model, lr, train_set, label_set, val_set=None):
    pos_weight = _get_pos_weight(label_set)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(pos_weight)
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    patience = 5
    best_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for seq, lbl in train_set:
            if torch.isnan(seq).any() or torch.isinf(seq).any():
                continue

            optimizer.zero_grad()

            output_flat, labels_flat = model.forward_train(seq, lbl)

            labels_flat = labels_flat.float()

            if labels_flat.shape != output_flat.shape:
                labels_flat = labels_flat.view_as(output_flat)

            loss = criterion(output_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_set)

        if val_set is not None:
            val_acc = eval_val_accuracy(model, val_set)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_model_state = copy.deepcopy(model.state_dict())

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if val_set is not None:
            print(f"Restored best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            print(f"Restored best model (Loss: {best_loss:.4f})")

def training_loop_sequential(epochs, model, lr, train_set, label_set, val_set=None, sampling_strategy=None):
    """
    Training loop for sequence models (Transformer). Each item in train_set is a full session.

    If val_set is provided:
        - Computes val accuracy every epoch
        - Early stops on val accuracy (patience)
        - Restores best model weights (by val acc) at the end
    Otherwise:
        - Falls back to early stopping on train loss

    If sampling_strategy is provided (for autoregressive models):
        - Computes per-epoch sampling probability and passes to forward_train
    """
    pos_weight = _get_pos_weight(label_set)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(pos_weight)
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_set)
    )

    patience = 5
    best_val_acc = 0
    best_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        sampling_prob = get_sampling_probability(sampling_strategy, epoch, epochs) if sampling_strategy else 0.0

        for seq, lbl in train_set:
            if torch.isnan(seq).any():
                continue

            optimizer.zero_grad()
            output_flat, labels_flat = model.forward_train(seq, lbl, sampling_prob)

            labels_flat = labels_flat.float()

            if labels_flat.shape != output_flat.shape:
                labels_flat = labels_flat.view_as(output_flat)

            loss = criterion(output_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_set)

        # Early stopping on training loss; best model saved by val acc (if available)
        if val_set is not None:
            val_acc = eval_val_accuracy(model, val_set)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_model_state = copy.deepcopy(model.state_dict())

        # Early stopping always tracks training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if val_set is not None:
            print(f"Restored best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            print(f"Restored best model (Loss: {best_loss:.4f})")

def testing_loop_independent(model, test_set):
    """Test batched independent models (DNN). Each item is a (batch_features, batch_labels) tuple."""
    model.eval()
    results_logger = Wave5ClassificationResultsLogger()
    freq_logger = FrequencyResults(Wave5ClassificationResultsLogger)

    with torch.no_grad():
        for inputs, labels in test_set:
            if torch.isnan(inputs).any():
                continue

            preds, logits = model.forward_test(inputs)
            labels = labels.float().view(-1)
            preds = preds.view(-1)

            results_logger.update_batch(preds, labels, logits)
            freq_logger.update_batch(
                preds=preds,
                labels=labels,
                logits=logits,
                freqs_log2=inputs[:, 0]
            )

    return results_logger, freq_logger


def testing_loop_sequential(model, test_set):
    model.eval()
    results_logger = Wave5ClassificationResultsLogger()
    seq_results_logger = SequenceResults(Wave5ClassificationResultsLogger)
    freq_results_logger = FrequencyResults(Wave5ClassificationResultsLogger)

    with torch.no_grad():
        for seq, label, total_len in test_set:
            if torch.isnan(seq).any():
                continue

            preds, logits = model.forward_test(seq)
            seq_results_logger.update(
                pred=preds,
                label=label,
                logits=logits,
                current_length=len(seq),
                total_length=total_len,
            )
            results_logger.update_sequential(preds, label, logits)
            freq_results_logger.update(
                pred=preds,
                label=label,
                logits=logits,
                freq_log2=seq[0][0].item()
            )

    return results_logger, seq_results_logger, freq_results_logger


def evaluate_grid_search(
        model_configs,
        model_factory,
        train_split,
        test_split,
        holdout_split,
        unique_labels,
        labels_categorical,
        sequential,
        root_dir,
        namespace):
    best_acc = 0
    i = 1
    plt_best = None
    title_text = None
    total = len(model_configs)

    random.shuffle(model_configs)

    for config in model_configs:
        ep = config.get("epochs", 25)
        lr = config.get("learning_rate", 0.001)
        sampling = config.get("sampling_strategy", "linear")

        print(f"{i} / {total}: Configuration: Epoch: {ep}, LR: {lr}, Sampling: {sampling}")
        i += 1

        model = model_factory(config)
        print_model_architecture(model, config)
        holdout_validation_used = config.get('holdout_validation', False)

        model_outputs = run_model(
            model=model,
            epochs=ep,
            lr=lr,
            train_set=train_split,
            test_set=test_split,
            label_set=labels_categorical,
            holdout_validation=holdout_validation_used,
            sequential=sequential,
            val_set=holdout_split,
            sampling_strategy=sampling
        )

        if holdout_validation_used:
            print("Holdout validation used, exporting results...")

        if sequential:
            results_logger, sequence_logger, freq_logger = model_outputs
        else:
            results_logger, freq_logger = model_outputs
            sequence_logger = None

        accuracy = results_logger.summarize_results()
        freq_logger.summarize_results()

        target_path = "holdout_validation" if holdout_validation_used else "train"
        save_path = f"{root_dir}/exported_features/wave_5/results/{target_path}/{namespace}/{namespace}_results.json"

        if accuracy > best_acc and is_higher_than_saved_acc(accuracy, save_path) or holdout_validation_used:
            print(f"Found better accuracy: {accuracy:.2f}%!")

            torch.save(model.state_dict(),
                       f'{root_dir}/exported_features/wave_5/weights/{namespace}_weights.pth')

            with open(f"{root_dir}/exported_features/wave_5/config/{namespace}_config.json", "w") as f:
                config_copy = config.copy()
                config_copy['holdout_validation'] = False
                json.dump(config_copy, f, indent=4)

            with open(save_path, "w") as f:
                json.dump(results_logger.get_results(), f, indent=4)

            with open(f"{root_dir}/exported_features/wave_5/results/{target_path}/{namespace}/freq_results.json",
                      "w") as f:
                json.dump(freq_logger.get_binned_results_dict(), f, indent=4)
            freq_logger.show_plot(
                title=f"Accuracy by Frequency ({namespace})",
                save_path=f"{root_dir}/exported_features/wave_5/results/{target_path}/{namespace}/freq_accuracy.png")

            if sequence_logger:
                sequence_logger.summarize_results()
                seq_save_path = f"{root_dir}/exported_features/wave_5/results/{target_path}/{namespace}/sequence_results.json"
                with open(seq_save_path, "w") as f:
                    json.dump(sequence_logger.get_binned_results_dict(), f, indent=4)

                sequence_logger.show_plot(
                    title=f"Accuracy vs Sequence Progression ({namespace})",
                    save_path=f"{root_dir}/exported_features/wave_5/results/{target_path}/{namespace}/sequence_accuracy.png")

            cm = results_logger.get_cm()

            plt_best = cm
            best_acc = accuracy
            title_text = f'Confusion Matrix ({namespace.upper()} Model) - {accuracy:.2f}'
            pred_df = results_logger.get_true_labels_preds()
            pred_df.to_csv(f"{root_dir}/exported_features/wave_5/results/{target_path}/{namespace}/pred_spread.csv", index=False)

            plot_cm(cm, title_text, 'Predicted Labels', 'True Labels', unique_labels,
                    save_path=f"{root_dir}/exported_features/wave_5/results/{target_path}/{namespace}/{namespace}_cm.png",
                    should_save=True, holdout_validation_cm=False)

    if plt_best is not None:
        print(f"Final Best Accuracy: {best_acc}")
        plot_cm(plt_best, title_text, 'Predicted Labels', 'True Labels', unique_labels,
                save_path=f"{root_dir}/exported_features/wave_5/results/train/{namespace}/{namespace}_best_cm.png",
                should_save=False, holdout_validation_cm=False)


current_dir = os.path.dirname(os.path.abspath(__file__))

config_root = os.path.join(current_dir, "../../../exported_features/wave_5/config")
config_root = os.path.abspath(config_root)

"""best_config_lstm = read_json_dict(os.path.join(config_root, "lstm_config.json"))
best_config_transformer = read_json_dict(os.path.join(config_root, "transformer_config.json"))"""
best_config_dnn = read_json_dict(os.path.join(config_root, "dnn_config.json"))

"""holdout_val_config_lstm = dict(best_config_lstm)
holdout_val_config_transformer = dict(best_config_transformer)"""
holdout_val_config_dnn = dict(best_config_dnn)
"""holdout_val_config_lstm['holdout_validation'] = True
holdout_val_config_transformer['holdout_validation'] = True"""
holdout_val_config_dnn['holdout_validation'] = True