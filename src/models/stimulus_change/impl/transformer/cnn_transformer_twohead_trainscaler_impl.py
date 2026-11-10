"""
Non-Autoregressive Transformer Two-Head — Train-Fitted Scaler variant
=====================================================================
This variant uses a fixed metadata scaler fit on the training set only:

- fit mean/std on training metadata only
- apply the same transform to train, val, and test
- waveform branch is unchanged (per-waveform demeaning inside the CNN)
- train uses full sequences
- val/test use prefix slices, scoring the last timestep
"""

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.models.stimulus_change.transformer.cnn_transformer.non_auto_transformer_model import NonAutoTransformerTwoHead
from src.utils.waveform_loader import WaveformSequenceDataset, load_waveform_data


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

requested_device = os.environ.get("ABR_DEVICE", "").strip().lower()
if requested_device:
    DEVICE = torch.device(requested_device)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

FIRST_HEAD_CLASSES = ["STOP", "REPEAT", "ADJUST"]
RESULTS_TAG = "non_auto_transformer_twohead_trainscaler_pca"


def create_freq_graph(values):
    value_counts = values.value_counts().sort_index()
    print(value_counts)
    print(f"Total {len(values)}")


def file_exists(path):
    return os.path.exists(path)


def plot_cm(cm, title, xlabel, ylabel, labels, save_path=None, should_save=False):
    plt.figure(figsize=(12, 10))
    vmax = np.percentile(cm, 99)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0, vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if should_save and save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def split_action_table(action_df):
    split_indices = action_df.index[action_df["stop"] == 1].tolist()
    prev = 0
    dfs = []
    for index in split_indices:
        dfs.append(action_df.iloc[prev:index + 1, :])
        prev = index + 1
    return dfs


def is_higher_than_saved_acc(accuracy, saved_results_path):
    if not file_exists(saved_results_path):
        return True
    with open(saved_results_path) as f:
        return accuracy > json.load(f)["accuracy"]


def ordinal_loss_action_head(action_logits, action_targets, action_class_to_db, scale=0.1):
    device = action_logits.device
    valid_mask = action_targets != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    valid_logits = action_logits[valid_mask]
    valid_targets = action_targets[valid_mask]
    probs = torch.softmax(valid_logits, dim=-1)

    acdb = action_class_to_db.to(device)
    db_valid = ~torch.isnan(acdb)
    db_values = acdb.clone()
    db_values[~db_valid] = 0

    valid_probs = probs * db_valid.float()
    prob_sum = valid_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized_probs = valid_probs / prob_sum
    expected_db = (normalized_probs * db_values).sum(dim=-1)

    true_db = acdb[valid_targets]
    ordinal_valid = ~torch.isnan(true_db)
    if not ordinal_valid.any():
        return torch.tensor(0.0, device=device)

    huber = torch.nn.functional.smooth_l1_loss(expected_db[ordinal_valid], true_db[ordinal_valid], reduction="mean")
    return scale * huber


def fit_train_meta_scaler(raw_sequences):
    """Fit metadata stats on the training set only.

    freq is left unchanged.
    last_level and extracted_1..10 are standardized with fixed train stats.
    """
    all_meta = np.concatenate([meta.detach().cpu().numpy() for _, meta, _ in raw_sequences], axis=0)

    last_level = all_meta[:, 1]
    last_mean = float(last_level.mean())
    last_std = float(last_level.std())

    pca_feats = all_meta[:, 2:12]
    pca_mean = pca_feats.mean(axis=0).astype(np.float32)
    pca_std = pca_feats.std(axis=0).astype(np.float32)

    return {
        "last_mean": last_mean,
        "last_std": last_std,
        "pca_mean": pca_mean,
        "pca_std": pca_std,
    }


def apply_train_meta_scaler(meta, stats, include_pca=True):
    meta = meta.copy()
    eps = 1e-8

    if stats["last_std"] > 0:
        meta[:, 1] = (meta[:, 1] - stats["last_mean"]) / (stats["last_std"] + eps)

    if include_pca and meta.shape[1] >= 12:
        nonzero = stats["pca_std"] > 0
        if nonzero.any():
            meta[:, 2:12][:, nonzero] = (
                meta[:, 2:12][:, nonzero] - stats["pca_mean"][nonzero]
            ) / (stats["pca_std"][nonzero] + eps)

    return meta


class TrainScaledSequenceDataset:
    """Wrap a waveform dataset and apply a fixed train-fitted scaler to metadata."""

    def __init__(self, base_dataset, scaler_stats):
        self.raw_sequences = base_dataset.raw_sequences
        self.sequences = []
        for wave, raw_meta, y in self.raw_sequences:
            raw_meta_np = raw_meta.detach().cpu().numpy()
            scaled_meta = apply_train_meta_scaler(raw_meta_np, scaler_stats, include_pca=(raw_meta_np.shape[1] >= 12))
            scaled_meta = torch.tensor(scaled_meta, dtype=raw_meta.dtype, device=raw_meta.device)
            self.sequences.append((wave, scaled_meta, y))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def __iter__(self):
        return iter(self.sequences)


def split_scaled_prefixes(sequence_dataset):
    records = []
    for wave, meta, y in sequence_dataset:
        for i in range(len(wave)):
            records.append((wave[:i + 1], meta[:i + 1], y[:i + 1]))
    return records


def compute_val_accuracy(model, val_prefixes, action_to_global):
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for wave, meta, label in val_prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)
            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue
            pred, _ = model.forward_test(wave, meta, action_to_global)
            correct += int(pred[-1].item() == label[-1].item())
            total += 1
    model.train()
    return correct / total * 100 if total > 0 else 0


def training_loop(epochs, model, lr, train_dataset, val_prefixes, stop_idx, repeat_idx, action_classes, global_to_action, action_to_global, action_class_to_db=None):
    device = next(model.parameters()).device

    all_train_labels = []
    for _, _, y in train_dataset:
        all_train_labels.extend(y.cpu().numpy().tolist())

    first_head_labels = [0 if lbl == stop_idx else 1 if lbl == repeat_idx else 2 for lbl in all_train_labels]
    first_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(first_head_labels))
    first_weights = torch.tensor(first_weights, dtype=torch.float32).to(device)
    first_criterion = nn.CrossEntropyLoss(weight=first_weights)

    action_train = [global_to_action[y] for y in all_train_labels if y != stop_idx and y != repeat_idx]
    present = np.unique(np.array(action_train))
    pw = compute_class_weight("balanced", classes=present, y=np.array(action_train))
    wmap = dict(zip(present, pw))
    fallback = float(np.median(pw))
    action_weights = np.array([wmap.get(c, fallback) for c in np.arange(action_classes)])
    action_weights = torch.tensor(action_weights, dtype=torch.float32).to(device)
    action_criterion = nn.CrossEntropyLoss(weight=action_weights, ignore_index=-100)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.98), eps=1e-9)
    steps_per_epoch = len(train_dataset)
    warmup_epochs = 5
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    cosine_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=2.5e-5),
        ],
        milestones=[warmup_steps],
    )

    print(f"  Train sequences: {len(train_dataset)} (full-sequence one-pass, train-fitted metadata scaler)")
    print(f"  Val prefixes: {len(val_prefixes)} (prefix slices, last timestep scored)")

    patience = 25
    best_val_acc = 0.0
    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve_loss = 0
    first_batch_printed = False

    for epoch in range(epochs):
        total_loss = total_first = total_action = total_ord = 0.0
        n_batches = 0
        model.train()
        for wave, meta, lbl in train_dataset:
            wave = wave.to(device)
            meta = meta.to(device)
            lbl = lbl.to(device)
            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            first_logits, action_logits, first_tgt, action_tgt = model.forward_train(wave, meta, lbl, global_to_action)
            first_loss = first_criterion(first_logits, first_tgt)
            if (action_tgt != -100).any():
                action_loss = action_criterion(action_logits, action_tgt)
                if torch.isnan(action_loss):
                    action_loss = torch.tensor(0.0, device=device)
            else:
                action_loss = torch.tensor(0.0, device=device)
            ord_loss = ordinal_loss_action_head(action_logits, action_tgt, action_class_to_db) if action_class_to_db is not None else torch.tensor(0.0, device=device)
            loss = first_loss + action_loss + ord_loss

            if not first_batch_printed:
                print(f"  [Debug] first_logits: {first_logits.shape}, action_logits: {action_logits.shape}")
                first_batch_printed = True

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_first += first_loss.item()
            total_action += action_loss.item()
            total_ord += ord_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_first = total_first / max(n_batches, 1)
        avg_action = total_action / max(n_batches, 1)
        avg_ord = total_ord / max(n_batches, 1)
        val_acc = compute_val_accuracy(model, val_prefixes, action_to_global)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f} (First: {avg_first:.4f}, Action: {avg_action:.4f}, Ord: {avg_ord:.4f}), Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")

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


def testing_loop(model, test_prefixes, reverse_map, action_to_global):
    device = next(model.parameters()).device
    model.eval()
    correct = total = pm10db_correct = pm5db_correct = pm15db_correct = quasi_total = top3_correct = top5_correct = 0
    predictions, true_labels = [], []
    with torch.no_grad():
        for wave, meta, label in test_prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)
            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue
            pred, last_logits = model.forward_test(wave, meta, action_to_global)
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
    print("Initializing data pipeline (Transformer Two-Head PCA — Train-Fitted Scaler)...")
    action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/action_table.csv")
    input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/input.csv")
    split_sessions = split_action_table(action_df)
    print(f"  Total sequences: {len(split_sessions)}")

    big_merged_df = pd.merge(input_df, action_df, on="record_id")
    label_ids = big_merged_df.iloc[:, 13:].idxmax(axis=1)
    create_freq_graph(label_ids)

    merged_dfs = [pd.merge(input_df, df, on="record_id") for df in split_sessions]
    merged_dfs = [df for df in merged_dfs if len(df) > 0]
    labels_cols = big_merged_df.iloc[:, 13:]
    mapping = {label: i for i, label in enumerate(labels_cols.columns)}
    reverse_mapping = {}
    for i, label in enumerate(labels_cols.columns):
        if "stop" in label:
            reverse_mapping[i] = "Stop"
        elif label == "0":
            reverse_mapping[i] = "Repeat"
        elif label[0] in "-<>":
            reverse_mapping[i] = f"{label} dB"
        else:
            reverse_mapping[i] = f"+{label} dB"
    labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]
    num_classes = len(mapping)
    print(f"Label Mapping Calculated!: {sorted(reverse_mapping.keys())}")

    stop_idx = mapping["stop"]
    repeat_idx = mapping["0"]
    action_classes = num_classes - 2
    global_to_action, action_to_global = {}, {}
    aid = 0
    for gid in range(num_classes):
        if gid in (stop_idx, repeat_idx):
            continue
        global_to_action[gid] = aid
        action_to_global[aid] = gid
        aid += 1

    index_to_colname = {v: k for k, v in mapping.items()}
    action_db_values = []
    for aid in range(action_classes):
        colname = index_to_colname[action_to_global[aid]]
        action_db_values.append(float("nan") if colname.startswith("<") or colname.startswith(">") else float(colname))
    action_class_to_db = torch.tensor(action_db_values, dtype=torch.float32)

    print("Loading waveform data...")
    waveform_df, record_to_bank = load_waveform_data()
    print(f"  Loaded {len(waveform_df)} waveforms")

    random.shuffle(merged_dfs)
    train_size = int(0.80 * len(merged_dfs))
    val_size = int(0.10 * len(merged_dfs))
    train_dfs = merged_dfs[:train_size]
    val_dfs = merged_dfs[train_size:train_size + val_size]
    test_dfs = merged_dfs[train_size + val_size:]

    if train_fraction < 1.0:
        train_dfs = train_dfs[:max(1, int(len(train_dfs) * train_fraction))]
    if val_fraction < 1.0:
        val_dfs = val_dfs[:max(1, int(len(val_dfs) * val_fraction))]

    base_train = WaveformSequenceDataset(train_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
    base_val = WaveformSequenceDataset(val_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
    base_test = WaveformSequenceDataset(test_dfs, mapping, waveform_df, record_to_bank, include_pca=True)

    scaler_stats = fit_train_meta_scaler(base_train.raw_sequences)
    train_dataset = TrainScaledSequenceDataset(base_train, scaler_stats)
    val_dataset = TrainScaledSequenceDataset(base_val, scaler_stats)
    test_dataset = TrainScaledSequenceDataset(base_test, scaler_stats)

    val_prefixes = split_scaled_prefixes(val_dataset)
    test_prefixes = split_scaled_prefixes(test_dataset)

    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")
    print(f"  Test sequences: {len(test_dataset)}")
    print(f"  Val prefixes: {len(val_prefixes)}")
    print(f"  Test prefixes: {len(test_prefixes)}")

    return {
        "mapping": mapping,
        "reverse_mapping": reverse_mapping,
        "labels": labels,
        "num_classes": num_classes,
        "stop_idx": stop_idx,
        "repeat_idx": repeat_idx,
        "action_classes": action_classes,
        "global_to_action": global_to_action,
        "action_to_global": action_to_global,
        "action_class_to_db": action_class_to_db,
        "train_dataset": train_dataset,
        "val_prefixes": val_prefixes,
        "test_prefixes": test_prefixes,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-fixed", action="store_true")
    parser.add_argument("--config", type=str, default=f"{ROOT_DIR}/exported_features/config/non_auto_transformer_twohead_pca_config.json")
    parser.add_argument("--weights", type=str, default=f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=1.0)
    args = parser.parse_args()

    data = build_datasets(train_fraction=args.train_fraction, val_fraction=args.val_fraction)

    with open(args.config) as f:
        config = json.load(f)
    config["output_size"] = data["num_classes"]
    config["stop_idx"] = data["stop_idx"]
    config["repeat_idx"] = data["repeat_idx"]
    config["action_classes"] = data["action_classes"]
    config["meta_dim"] = 12

    model = NonAutoTransformerTwoHead(config).to(DEVICE)

    if args.eval:
        model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=DEVICE))
        accuracy, cm, results = testing_loop(model, data["test_prefixes"], data["reverse_mapping"], data["action_to_global"])
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        plot_cm(cm, f"Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%", "Predicted Labels", "True Labels", data["labels"])
    else:
        print(f"\n{'='*60}")
        print(f"TRAIN MODE: {RESULTS_TAG}")
        print("=" * 60)
        training_loop(
            epochs=args.epochs,
            model=model,
            lr=5e-5,
            train_dataset=data["train_dataset"],
            val_prefixes=data["val_prefixes"],
            stop_idx=data["stop_idx"],
            repeat_idx=data["repeat_idx"],
            action_classes=data["action_classes"],
            global_to_action=data["global_to_action"],
            action_to_global=data["action_to_global"],
            action_class_to_db=data["action_class_to_db"],
        )
        accuracy, cm, results = testing_loop(model, data["test_prefixes"], data["reverse_mapping"], data["action_to_global"])
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        results_path = f"{ROOT_DIR}/exported_features/results/train/transformer/{RESULTS_TAG}_results.json"
        if is_higher_than_saved_acc(accuracy, results_path):
            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/transformer", exist_ok=True)
            torch.save(model.state_dict(), f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print("Saved new best model!")
        plot_cm(cm, f"Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%", "Predicted Labels", "True Labels", data["labels"])
