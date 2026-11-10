"""
Hybrid CNN-LSTM — Causal Running-Norm variant
=============================================
Single-head model with one-pass causal metadata normalization:

- timestep t is normalized using only timesteps 0..t
- training remains one forward pass per full sequence
- validation/test use prefix slices of the causally-normalized sequence

This gives prefix-style normalization behavior without expanding training into
all prefixes.
"""

import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

from src.models.stimulus_change.impl.lstm.hybrid_cnn_lstm_impl import (
    DEVICE,
    ROOT_DIR,
    HybridCNNLSTMModel,
    class_to_db,
    create_freq_graph,
    file_exists,
    get_sampling_probability,
    is_higher_than_saved_acc,
    load_waveform_data,
    mapping,
    ordinal_distance_loss,
    plot_cm,
    print_model_architecture,
    reverse_mapping,
    split_action_table,
    labels,
    model_configs,
)
from src.utils.waveform_loader import WaveformSequenceDataset

import pandas as pd


RESULTS_TAG = "hybrid_cnn_lstm_runningnorm"


def causal_running_normalize_meta(meta, include_pca=True):
    meta = meta.copy()
    out = meta.copy()
    eps = 1e-8

    running_sum = 0.0
    running_sumsq = 0.0
    for t, value in enumerate(meta[:, 1]):
        running_sum += float(value)
        running_sumsq += float(value) * float(value)
        count = t + 1
        mean = running_sum / count
        var = max(running_sumsq / count - mean * mean, 0.0)
        std = np.sqrt(var)
        if std > 0:
            out[t, 1] = (value - mean) / (std + eps)

    if include_pca and meta.shape[1] >= 12:
        running_sum = np.zeros(10, dtype=np.float64)
        running_sumsq = np.zeros(10, dtype=np.float64)
        for t in range(meta.shape[0]):
            values = meta[t, 2:12].astype(np.float64)
            running_sum += values
            running_sumsq += values * values
            count = t + 1
            mean = running_sum / count
            var = np.maximum(running_sumsq / count - mean * mean, 0.0)
            std = np.sqrt(var)
            nonzero = std > 0
            normalized = values.copy()
            normalized[nonzero] = (values[nonzero] - mean[nonzero]) / (std[nonzero] + eps)
            out[t, 2:12] = normalized.astype(np.float32)

    return out


class RunningNormSequenceDataset:
    def __init__(self, base_dataset):
        self.raw_sequences = base_dataset.raw_sequences
        self.sequences = []
        for wave, raw_meta, y in self.raw_sequences:
            raw_meta_np = raw_meta.detach().cpu().numpy()
            running_meta = causal_running_normalize_meta(raw_meta_np, include_pca=(raw_meta_np.shape[1] >= 12))
            running_meta = torch.tensor(running_meta, dtype=raw_meta.dtype, device=raw_meta.device)
            self.sequences.append((wave, running_meta, y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def __iter__(self):
        return iter(self.sequences)


def split_runningnorm_prefixes(sequence_dataset):
    records = []
    for wave, meta, y in sequence_dataset:
        for i in range(len(wave)):
            records.append((wave[:i + 1], meta[:i + 1], y[:i + 1]))
    return records


def compute_val_accuracy(model, val_prefixes):
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for wave, meta, label in val_prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)
            pred, _ = model.forward_test(wave, meta)
            correct += int(pred[-1].item() == label[-1].item())
            total += 1
    model.train()
    return correct / total * 100 if total > 0 else 0


def training_loop(epochs, model, lr, train_dataset, val_prefixes, label_map, sampling_strategy, class_to_db_tensor):
    device = next(model.parameters()).device

    all_train_labels = []
    for _, _, y in train_dataset:
        all_train_labels.extend(y.cpu().numpy().tolist())

    class_weights = compute_class_weight("balanced", classes=np.array(list(label_map.values())), y=np.array(all_train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    class_to_db_tensor = class_to_db_tensor.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    steps_per_epoch = len(train_dataset)
    warmup_epochs = 20
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6),
        ],
        milestones=[warmup_steps],
    )

    print(f"  Train sequences: {len(train_dataset)} (full-sequence one-pass, causal running norm)")
    print(f"  Val prefixes: {len(val_prefixes)} (prefix slices, score last timestep)")

    patience = 25
    best_val_acc = 0.0
    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve_loss = 0

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        model.train()

        for wave, meta, lbl in train_dataset:
            wave = wave.to(device)
            meta = meta.to(device)
            lbl = lbl.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            optimizer.zero_grad()
            sampling_prob = get_sampling_probability(sampling_strategy, epoch, epochs)
            output_flat, labels_flat, decorr_loss = model.forward_train(wave, meta, lbl, sampling_prob)
            ce_loss = criterion(output_flat, labels_flat)
            ord_loss = ordinal_distance_loss(output_flat, labels_flat, class_to_db_tensor, scale=0.03)
            loss = ce_loss + ord_loss + 0.05 * decorr_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_acc = compute_val_accuracy(model, val_prefixes)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best model saved! (val acc: {val_acc:.2f}%)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve_loss = 0
        else:
            epochs_no_improve_loss += 1

        if epochs_no_improve_loss >= patience:
            print(f"Early stopping at epoch {epoch+1} (loss stopped improving)")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val acc: {best_val_acc:.2f}%)")


def testing_loop(model, test_prefixes, reverse_map):
    device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    pm10db_correct, pm5db_correct, pm15db_correct, quasi_total = 0, 0, 0, 0
    top3_correct, top5_correct = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for wave, meta, label in test_prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            pred, last_logits = model.forward_test(wave, meta)
            pred_last = pred[-1].item()
            label_last = label[-1].item()
            correct += int(pred_last == label_last)
            total += 1

            correct_mapped = reverse_map[label_last]
            pred_mapped = reverse_map[pred_last]
            top3 = torch.topk(last_logits, k=3).indices.tolist()
            top5 = torch.topk(last_logits, k=5).indices.tolist()
            top3_correct += int(label_last in top3)
            top5_correct += int(label_last in top5)

            if ("Stop" not in correct_mapped and "Stop" not in pred_mapped and ">" not in correct_mapped and ">" not in pred_mapped and "<" not in correct_mapped and "<" not in pred_mapped):
                quasi_total += 1
                value = "0" if "Repeat" in correct_mapped else correct_mapped.split(" ")[0]
                value_pred = "0" if "Repeat" in pred_mapped else pred_mapped.split(" ")[0]
                if abs(int(value) - int(value_pred)) <= 5:
                    pm5db_correct += 1
                if abs(int(value) - int(value_pred)) <= 10:
                    pm10db_correct += 1
                if abs(int(value) - int(value_pred)) <= 15:
                    pm15db_correct += 1

            predictions.append(pred_last)
            true_labels.append(label_last)

    accuracy = correct / total * 100
    top3_acc = top3_correct / total * 100
    top5_acc = top5_correct / total * 100

    print(f"\nFraction: {correct}/{total}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    if quasi_total > 0:
        print(f"Within 10 dB: {pm10db_correct}/{quasi_total} = {100*(pm10db_correct/quasi_total):.2f}%")
        print(f"Within 5 dB: {pm5db_correct}/{quasi_total} = {100*(pm5db_correct/quasi_total):.2f}%")
        print(f"Within 15 dB: {pm15db_correct}/{quasi_total} = {100*(pm15db_correct/quasi_total):.2f}%")

    results = {
        "accuracy": accuracy,
        "top3_acc": top3_acc,
        "top5_acc": top5_acc,
        "pm10db_correct": 100 * (pm10db_correct / quasi_total) if quasi_total > 0 else 0,
        "pm5db_correct": 100 * (pm5db_correct / quasi_total) if quasi_total > 0 else 0,
        "pm15db_correct": 100 * (pm15db_correct / quasi_total) if quasi_total > 0 else 0,
    }

    cm = confusion_matrix(y_true=true_labels, y_pred=predictions)
    return accuracy, cm, results


def build_datasets(train_fraction=1.0, val_fraction=1.0):
    print("Initializing data pipeline (Hybrid CNN-LSTM — Causal Running Norm)...")
    action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/action_table.csv")
    input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/input.csv")

    split_sessions = split_action_table(action_df)
    action_df_clean = pd.concat(split_sessions, ignore_index=True)
    big_merged_df = pd.merge(input_df, action_df_clean, on="record_id")
    label_ids = big_merged_df.iloc[:, 13:].idxmax(axis=1)
    create_freq_graph(label_ids)

    merged_dfs = [pd.merge(input_df, df, on="record_id") for df in split_sessions]
    merged_dfs = [df for df in merged_dfs if len(df) > 0]

    random.shuffle(merged_dfs)
    train_size = int(0.8 * len(merged_dfs))
    val_size = int(0.1 * len(merged_dfs))

    train_dfs = merged_dfs[:train_size]
    val_dfs = merged_dfs[train_size:train_size + val_size]
    test_dfs = merged_dfs[train_size + val_size:]

    if train_fraction < 1.0:
        train_dfs = train_dfs[:max(1, int(len(train_dfs) * train_fraction))]
    if val_fraction < 1.0:
        val_dfs = val_dfs[:max(1, int(len(val_dfs) * val_fraction))]

    waveform_df, record_to_bank = load_waveform_data()

    base_train = WaveformSequenceDataset(train_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
    base_val = WaveformSequenceDataset(val_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
    base_test = WaveformSequenceDataset(test_dfs, mapping, waveform_df, record_to_bank, include_pca=True)

    train_dataset = RunningNormSequenceDataset(base_train)
    val_dataset = RunningNormSequenceDataset(base_val)
    test_dataset = RunningNormSequenceDataset(base_test)

    val_prefixes = split_runningnorm_prefixes(val_dataset)
    test_prefixes = split_runningnorm_prefixes(test_dataset)

    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")
    print(f"  Test sequences: {len(test_dataset)}")
    print(f"  Val prefixes: {len(val_prefixes)}")
    print(f"  Test prefixes: {len(test_prefixes)}")

    return train_dataset, val_prefixes, test_prefixes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-fixed", action="store_true")
    parser.add_argument("--config", type=str, default=f"{ROOT_DIR}/exported_features/config/hybrid_cnn_lstm_config.json")
    parser.add_argument("--weights", type=str, default=f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    args = parser.parse_args()

    train_dataset, val_prefixes, test_prefixes = build_datasets(args.train_fraction, args.val_fraction)

    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = dict(model_configs[0])
        print(f"Config not found at {args.config}; using built-in fixed config fallback.")

    config["output_size"] = len(mapping)
    config["meta_dim"] = 12

    model = HybridCNNLSTMModel(config).to(DEVICE)
    print_model_architecture(model, config)

    if args.eval:
        model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=DEVICE))
        accuracy, cm, results = testing_loop(model, test_prefixes, reverse_mapping)
    else:
        training_loop(
            epochs=args.epochs,
            model=model,
            lr=0.0005,
            train_dataset=train_dataset,
            val_prefixes=val_prefixes,
            label_map=mapping,
            sampling_strategy="linear",
            class_to_db_tensor=class_to_db,
        )
        accuracy, cm, results = testing_loop(model, test_prefixes, reverse_mapping)
        results_path = f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_results.json"
        if is_higher_than_saved_acc(accuracy, results_path):
            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm", exist_ok=True)
            torch.save(model.state_dict(), f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print("Saved new best model!")

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    plot_cm(cm, f"Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%", "Predicted Labels", "True Labels", labels)
