import copy

import numpy as np
from sklearn.utils import compute_class_weight
from torch import nn, optim

from src.common.model.model_util_common import *
from src.common.model.results.aggregator.freq_results import FrequencyResults
from src.common.model.results.aggregator.sequence_results import SequenceResults
from src.common.model.results.logger.stimulus_change_results import StimulusChangeResults
from src.common.model.api.sequential_model import ConfigurableSequentialModel
from src.common.plot_util import plot_cm, create_freq_graph


def get_sampling_probability(type, epoch, epochs):
    if type == "linear":
        return min(0.5, epoch / epochs)
    if type == "constant":
        return 0.3
    else:
        return 0


def training_loop(epochs, model, lr, train_set, label_map, sampling_strategy, label_set, val_set=None):
    class_weights = compute_class_weight('balanced', classes=np.array(list(label_map.values())), y=label_set)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,  # peak LR
        epochs=epochs,
        steps_per_epoch=len(train_set)
    )

    patience = 5
    best_loss = float('inf')
    best_val_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for seq, lbl in train_set:
            if torch.isnan(seq).any():
                continue
            optimizer.zero_grad()
            sampling_prob = get_sampling_probability(sampling_strategy, epoch, epochs)
            output_flat, labels_flat = model.forward_train(seq, lbl, sampling_prob)
            loss = criterion(output_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
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

        # Track early stopping based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # --- Restore Best Weights ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if val_set is not None:
            print(f"Restored best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            print(f"Restored best model (Loss: {best_loss:.4f})")

# test set is a list of tuples of sequences and labels
def testing_loop_sequential(model, test_set, reverse_map):
    model.eval()
    results_logger = StimulusChangeResults()
    freq_logger = FrequencyResults(StimulusChangeResults)
    seq_logger = SequenceResults(StimulusChangeResults, num_bins=10)

    with torch.no_grad():
        for seq, label, total_len in test_set:
            if torch.isnan(seq).any():
                continue

            pred, last_logits = model.forward_test(seq)

            results_logger.update_sequential(pred, label, last_logits, reverse_map=reverse_map)
            seq_logger.update(
                pred=pred,
                label=label,
                logits=last_logits,
                current_length=len(seq),
                total_length=total_len,
                reverse_map=reverse_map
            )

            freq_logger.update(
                pred=pred,
                label=label,
                logits=last_logits,
                freq_log2=seq[0][0].item(),
                reverse_map=reverse_map
            )

    return results_logger, freq_logger, seq_logger

def testing_loop_independent(model, test_set, reverse_map):
    model.eval()
    results_logger = StimulusChangeResults()
    freq_logger = FrequencyResults(StimulusChangeResults)

    with torch.no_grad():
        for inputs, labels in test_set:
            if torch.isnan(inputs).any():
                continue
            preds, logits = model.forward_test(inputs)
            results_logger.update_batch(preds=preds, labels=labels, logits=logits, reverse_map=reverse_map)
            freq_logger.update_batch(
                preds=preds,
                labels=labels,
                logits=logits,
                freqs_log2=inputs[:, 0],
                reverse_map=reverse_map
            )
    return results_logger, freq_logger

# label map maps -20 -> 1, -40 -> 2
# reverse maps 0 -> 10, 7 -> 0, etc
# label set is the set of all labels (mapped)
def run_model(model, epochs, lr, sampling_strategy, train_set, test_set, holdout_set, label_map, reverse_map, label_set, holdout_validation):
    if holdout_validation:
        print("Using holdout validation training, merging train and test set")
        train_set = (train_set + test_set)
        test_set = holdout_set
        holdout_set = None

    # Pass val_set down to the training loop
    training_loop(epochs=epochs, model=model, lr=lr, train_set=train_set,
                  label_map=label_map, sampling_strategy=sampling_strategy,
                  label_set=label_set, val_set=holdout_set)

    if isinstance(model, ConfigurableSequentialModel):
        return testing_loop_sequential(model, split_indiv_records(test_set), reverse_map)
    return testing_loop_independent(model, test_set, reverse_map)

def get_seq_split_inputs_outputs(action_df, input):
    action_df.insert(1, 'sequence_id', action_df['stop'].shift(fill_value=0).cumsum())

    big_merged_df = pd.merge(left=input, right=action_df, on="record_id", how='left')
    label_ids = big_merged_df.iloc[:, 49:].idxmax(axis=1)
    create_freq_graph(label_ids)

    merged_dfs = [
        group.drop(columns=['sequence_id'])
        for _, group in big_merged_df.groupby('sequence_id')
        if not group.empty
    ]

    action_df.drop(columns=['sequence_id'], inplace=True)
    big_merged_df.drop(columns=['sequence_id'], inplace=True)

    return merged_dfs, big_merged_df, label_ids

def get_inputs_outputs(input, action_df):
    big_merged_df = pd.merge(left=input, right=action_df, on="record_id", how='left')
    label_ids = big_merged_df.iloc[:, 48:]
    label_ids = label_ids.idxmax(axis=1)
    create_freq_graph(label_ids)
    input_data = big_merged_df.iloc[:, :47]
    return input_data, big_merged_df, label_ids

def get_label_mappings(big_merged_df, idx=48):
    labels = big_merged_df.iloc[:, idx:]
    mapping = {label: i for i, label in enumerate(labels.columns)}
    reverse_mapping = {v: k for k, v in sorted(mapping.items())}
    temp = {}
    for key, value in reverse_mapping.items():
        if "stop" in value:
            temp[key] = value.capitalize()
        elif value == "0":
            temp[key] = "Repeat"
        else:
            if value[0] == '-':
                temp[key] = '-' + value[1:] + " dB"
            elif value[0] == '>' or value[0] == '<':
                temp[key] = value + " dB"
            else:
                temp[key] = "+" + value + " dB"
    reverse_mapping = temp
    return mapping, reverse_mapping

def split_df_data_labels(df, label_map):
    df_labels = df.iloc[:, 13:].idxmax(axis=1)
    df_labels = df_labels.map(label_map).to_numpy()
    x = torch.tensor(df.iloc[:, 1:13].values, dtype=torch.float32)
    y = torch.tensor(df_labels, dtype=torch.int64)
    return x, y

def evaluate_grid_search(
        model_configs,
        model_factory,
        train_split,
        test_split,
        holdout_split,
        mapping,
        reverse_mapping,
        unique_labels,
        labels_categorical,
        root_dir,
        namespace):
    global target_path
    best_acc = 0
    i = 1
    plt_best = None
    title_text = None
    total = len(model_configs)

    random.shuffle(model_configs)

    for config in model_configs:
        ep = config.get("epochs", 25)
        lr = config.get("learning_rate", 0.01)
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
            sampling_strategy=sampling,
            train_set=train_split,
            test_set=test_split,
            holdout_set=holdout_split,
            label_map=mapping,
            reverse_map=reverse_mapping,
            label_set=labels_categorical,
            holdout_validation=holdout_validation_used
        )
        has_sequence_logger = len(model_outputs) == 3
        if has_sequence_logger:
            results_logger, freq_logger, sequence_logger = model_outputs
        else:
            results_logger, freq_logger = model_outputs
            sequence_logger = None

        if holdout_validation_used:
            print("Holdout validation used, exporting results...")

        accuracy = results_logger.summarize_results()
        freq_logger.summarize_results()
        target_path = "holdout_validation" if holdout_validation_used else "train"
        save_path = f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/{namespace}_results.json"

        if ((accuracy > best_acc and is_higher_than_saved_acc(accuracy, save_path)) or holdout_validation_used) and not results_logger.is_overpredicting():
            print(f"Found better accuracy: {accuracy:.2f}%!")

            with open(f"{root_dir}/exported_features/stimulus_change/config/{namespace}_config.json", "w") as f:
                config['holdout_validation'] = False
                json.dump(config, f, indent=4)

            with open(save_path, "w") as f:
                json.dump(results_logger.get_results(), f, indent=4)
            with open(f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/freq_results.json", "w") as f:
                json.dump(freq_logger.get_binned_results_dict(), f, indent=4)
            freq_logger.show_plot(title=f"Accuracy by Frequency ({namespace})",
                                  save_path=f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/freq_accuracy.png")

            if has_sequence_logger:
                sequence_logger.summarize_results()
                seq_save_path = f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/sequence_results.json"
                with open(seq_save_path, "w") as f:
                    json.dump(sequence_logger.get_binned_results_dict(), f, indent=4)
                sequence_logger.show_plot(
                    title=f"Accuracy vs Sequence Progression ({namespace})",
                    save_path=f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/sequence_accuracy.png")

            cm = results_logger.get_cm()
            pred_df = results_logger.get_true_labels_preds()
            pred_df.to_csv(f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/pred_spread.csv", index=False)

            with open(f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/reverse_mapping.json", "w") as f:
                json.dump(reverse_mapping, f, indent=4)

            plt_best = cm
            best_acc = accuracy
            title_text = f'Confusion Matrix ({namespace.upper()} Model) - {accuracy:.2f}'

            plot_cm(cm, title_text, 'Predicted Labels', 'True Labels', unique_labels,
                    save_path=f"{root_dir}/exported_features/stimulus_change/results/{target_path}/{namespace}/{namespace}_cm.png",
                    should_save=True, holdout_validation_cm=holdout_validation_used)

    if plt_best is not None:
        print(f"Final Best Accuracy: {best_acc}")
        plot_cm(plt_best, title_text, 'Predicted Labels', 'True Labels', unique_labels,
                save_path=f"{root_dir}/exported_features/stimulus_change/results/train/{namespace}/{namespace}_best_cm.png",
                should_save=False, holdout_validation_cm=False)



current_dir = os.path.dirname(os.path.abspath(__file__))

config_root = os.path.join(current_dir, "../../../exported_features/stimulus_change/config")
config_root = os.path.abspath(config_root)

best_config_lstm = read_json_dict(os.path.join(config_root, "lstm_config.json"))
best_config_transformer = read_json_dict(os.path.join(config_root, "transformer_config.json"))
best_config_dnn = read_json_dict(os.path.join(config_root, "dnn_config.json"))
best_config_random = read_json_dict(os.path.join(config_root, "random_config.json"))

holdout_val_config_lstm = dict(best_config_lstm)
holdout_val_config_transformer = dict(best_config_transformer)
holdout_val_config_dnn = dict(best_config_dnn)
holdout_val_config_lstm['holdout_validation'] = True
holdout_val_config_transformer['holdout_validation'] = True
holdout_val_config_dnn['holdout_validation'] = True