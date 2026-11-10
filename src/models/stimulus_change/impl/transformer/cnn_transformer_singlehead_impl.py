"""
Non-Autoregressive Transformer Single-Head — PCA variant (ALL classes)
======================================================================
NON-AUTOREGRESSIVE: no previous-action feedback, no compounding errors.
The model sees CNN(waveform) + meta (freq, last_level, PCA) at each step,
uses causal masking so step t only sees steps 0..t, and predicts every action
in a single forward pass.

Single-head model: direct multi-class classification over ALL action classes.
NO rare class dropping — keeps -25, -15, +15, and everything else.

PCA ON → meta_dim = 12 (freq + last_level + 10 extracted)

Usage:
    cd src/models/impl
    python non_auto_transformer_pca_impl.py
    python non_auto_transformer_pca_impl.py --train-fixed
    python non_auto_transformer_pca_impl.py --eval
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

# Set seeds for reproducibility (CRITICAL for consistent train/test splits!)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Auto-detect best available device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.waveform_loader import (
    load_waveform_data,
    WaveformSequenceDataset,
    split_for_testing_waveform
)
from src.models.stimulus_change.transformer.cnn_transformer.non_auto_transformer_singlehead_model import NonAutoTransformerSingleHead


# ============================================================================
# Helper functions (copied to avoid psycopg2/data_util dependency)
# ============================================================================

def create_freq_graph(values):
    """Create frequency distribution graph (non-blocking)."""
    value_counts = values.value_counts().sort_index()
    print(value_counts)
    print(f'Total {len(values)}')

def file_exists(path):
    return os.path.exists(path)

def plot_cm(cm, title, xlabel, ylabel, labels, save_path=None, should_save=False):
    """Plot confusion matrix."""
    plt.figure(figsize=(12, 10))
    vmax = np.percentile(cm, 99)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if should_save and save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def split_action_table(action_df):
    split_indices = action_df.index[action_df['stop'] == 1].tolist()
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
        results = json.load(f)
        prev_acc = results["accuracy"]
        return accuracy > prev_acc


# ============================================================================
# Validation — uses fixed prefixes (same protocol as LSTM / two-head)
# ============================================================================

def create_fixed_val_prefixes(val_data):
    """Create fixed validation prefixes — each sequence gets one of 25%, 50%, 75%, 100%."""
    prefixes = []
    percentages = [0.25, 0.50, 0.75, 1.0]
    for i, (wave, meta, label) in enumerate(val_data):
        T = wave.size(0)
        pct = percentages[i % 4]
        t = max(1, int(T * pct))
        prefixes.append((wave[:t], meta[:t], label[:t]))
    return prefixes


def compute_val_accuracy(model, val_prefixes):
    """Validation accuracy using fixed prefixes."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for wave, meta, label in val_prefixes:
            device = next(model.parameters()).device
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)
            pred, _ = model.forward_test(wave, meta)
            correct += int(pred[-1].item() == label[-1].item())
            total += 1
    model.train()
    return correct / total * 100 if total > 0 else 0


# ============================================================================
# Training Loop (Non-Autoregressive — no scheduled sampling)
# ============================================================================

def training_loop(epochs, model, lr, train_dataset, val_dataset, label_map,
):
    """Non-autoregressive single-head training with all-timestep loss."""
    device = next(model.parameters()).device

    # --- Class weights (balanced) ---
    all_train_labels = []
    for _, _, y in train_dataset:
        all_train_labels.extend(y.cpu().numpy().tolist())
    train_labels_np = np.array(all_train_labels)

    class_ids = np.array(sorted(label_map.values()))
    class_weights = compute_class_weight('balanced', classes=class_ids, y=train_labels_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Warmup for 10 epochs, then cosine decay
    # (Transformers are more LR-sensitive than LSTMs — shorter warmup, lower peak)
    steps_per_epoch = len(train_dataset)
    warmup_epochs = 10
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=2.5e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    val_prefixes = create_fixed_val_prefixes(val_dataset)
    print(f"  Created {len(val_prefixes)} fixed val prefixes (25%/50%/75%/100% spread)")

    patience = 25
    best_val_acc = 0.0
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve_loss = 0
    first_batch_printed = False

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        model.train()

        for seq_idx, (wave, meta, lbl) in enumerate(train_dataset):
            wave = wave.to(device)
            meta = meta.to(device)
            lbl = lbl.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            # Non-autoregressive: predict ALL timesteps at once
            logits, targets = model.forward_train(wave, meta, lbl)

            if not first_batch_printed:
                print(f"  [Debug] logits: {logits.shape}, targets: {targets.shape}")
                first_batch_printed = True

            loss = criterion(logits, targets)

            loss.backward()
            total_loss += loss.item()
            n_batches += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)

        val_acc = compute_val_accuracy(model, val_prefixes)

        current_lr = optimizer.param_groups[0]['lr']
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


# ============================================================================
# Testing Loop
# ============================================================================

def testing_loop(model, test_set, reverse_map):
    device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    pm10db_correct, pm5db_correct, pm15db_correct, quasi_total = 0, 0, 0, 0
    top3_correct, top5_correct = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for wave, meta, label in test_set:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            pred, last_logits = model.forward_test(wave, meta)
            correct += (1 if pred[-1].item() == label[-1].item() else 0)
            total += 1

            correct_mapped = reverse_map[label[-1].item()]
            pred_mapped = reverse_map[pred[-1].item()]

            top3 = torch.topk(last_logits, k=3).indices.tolist()
            top5 = torch.topk(last_logits, k=5).indices.tolist()
            top3_correct += int(label[-1].item() in top3)
            top5_correct += int(label[-1].item() in top5)

            if ('Stop' not in correct_mapped and 'Stop' not in pred_mapped and
                '>' not in correct_mapped and '>' not in pred_mapped and
                '<' not in correct_mapped and '<' not in pred_mapped):
                quasi_total += 1
                value = correct_mapped.split(" ")[0]
                value_pred = pred_mapped.split(" ")[0]
                if "Repeat" in value:
                    value = "0"
                if "Repeat" in value_pred:
                    value_pred = "0"
                if abs(int(value) - int(value_pred)) <= 5:
                    pm5db_correct += 1
                if abs(int(value) - int(value_pred)) <= 15:
                    pm15db_correct += 1
                if abs(int(value) - int(value_pred)) <= 10:
                    pm10db_correct += 1

            predictions.append(pred[-1].item())
            true_labels.append(label[-1].item())

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
        "pm10db_correct": 100*(pm10db_correct / quasi_total) if quasi_total > 0 else 0,
        "pm5db_correct": 100*(pm5db_correct / quasi_total) if quasi_total > 0 else 0,
        "pm15db_correct": 100*(pm15db_correct / quasi_total) if quasi_total > 0 else 0,
    }

    cm = confusion_matrix(y_true=true_labels, y_pred=predictions)
    return accuracy, cm, results


def run_model(model, epochs, lr, train_dataset, val_dataset, test_set,
              label_map, reverse_map):
    training_loop(
        epochs=epochs,
        model=model,
        lr=lr,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        label_map=label_map,
    )
    return testing_loop(model, test_set, reverse_map)


def print_model_architecture(model, config):
    print(f" CNN embed dim         : {config.get('cnn_embed_dim', 96)}")
    print(f" CNN pool_k            : {config.get('pool_k', 8)}")
    print(f" Meta dim              : {config.get('meta_dim', 12)} (freq+level+extracted)")
    print(f" Output size           : {config['output_size']} (ALL classes, nothing dropped)")
    print(f" Transformer dim       : {config['transformer_dim']}")
    print(f" Transformer heads     : {config['transformer_heads']}")
    print(f" Transformer FF        : {config['transformer_ff']}")
    print(f" Transformer layers    : {config['transformer_layers']}")
    print(f" Dropout               : {config.get('dropout', 0.2)}")
    print(f" Activation            : {config.get('activation', 'relu')}")
    print(f" Pre-FC Layers         : {config.get('pre_fc_layers', [])}")
    print(f" Post-FC Layers        : {config.get('post_fc_layers', [])}")
    print(f" Mode                  : NON-AUTOREGRESSIVE single-head (no prev action feedback)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTOTAL PARAMETERS: {total_params:,}")


# ============================================================================
# Data Loading — ALL CLASSES (no rare class dropping)
# ============================================================================

import pandas as pd

print("Initializing data pipeline (Non-Auto Transformer Single-Head PCA — ALL classes)...")
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/action_table.csv")
input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/input.csv")

# NO rare class dropping — keep everything
split_sessions = split_action_table(action_df)
print(f"  Total sequences: {len(split_sessions)} (ALL classes kept, nothing dropped)")

action_df_clean = action_df  # use all data
big_merged_df = pd.merge(input_df, action_df_clean, on="record_id")
label_ids = big_merged_df.iloc[:, 13:].idxmax(axis=1)
create_freq_graph(label_ids)

merged_dfs = [pd.merge(input_df, df, on='record_id') for df in split_sessions]
merged_dfs = [df for df in merged_dfs if len(df) > 0]

labels_cols = big_merged_df.iloc[:, 13:]
mapping = {label: i for i, label in enumerate(labels_cols.columns)}
reverse_mapping = {}
for i, label in enumerate(labels_cols.columns):
    if "stop" in label:
        reverse_mapping[i] = "Stop"
    elif label == "0":
        reverse_mapping[i] = "Repeat"
    elif label[0] == '-':
        reverse_mapping[i] = f"{label} dB"
    elif label[0] == '>' or label[0] == '<':
        reverse_mapping[i] = f"{label} dB"
    else:
        reverse_mapping[i] = f"+{label} dB"

labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]
labels_categorical = np.array(label_ids.map(mapping))

num_classes = len(mapping)
print(f"Label Mapping Calculated!: {sorted(reverse_mapping.keys())} ({num_classes} classes)")

# Load waveform data
print("Loading waveform data...")
waveform_df, record_to_bank = load_waveform_data()
print(f"  Loaded {len(waveform_df)} waveforms")

# Split: 80-10-10
print("Preparing sequences (HYBRID mode: CNN + PCA, include_pca=True)...")
print("Splitting into 80-10-10...")
random.shuffle(merged_dfs)
train_size = int(0.8 * len(merged_dfs))
val_size = int(0.1 * len(merged_dfs))

train_dfs = merged_dfs[:train_size]
val_dfs = merged_dfs[train_size:train_size + val_size]
test_dfs = merged_dfs[train_size + val_size:]

# PCA ON → include_pca=True, meta_dim=12
train_dataset = WaveformSequenceDataset(train_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
val_dataset = WaveformSequenceDataset(val_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
test_dataset = WaveformSequenceDataset(test_dfs, mapping, waveform_df, record_to_bank, include_pca=True)

print(f"  Train sequences: {len(train_dataset)} (80%)")
print(f"  Val sequences: {len(val_dataset)} (10%)")
print(f"  Test sequences: {len(test_dataset)} (10%)")

test_split_prev_preds = split_for_testing_waveform(test_dataset)
print(f"  Test prefixes: {len(test_split_prev_preds)}")

if len(train_dataset) > 0:
    wave, meta, y = train_dataset[0]
    print(f"\n  Sample shapes (PCA ON):")
    print(f"    wave: {wave.shape}")
    print(f"    meta: {meta.shape}")  # [T, 12]
    print(f"    y: {y.shape}")


# ============================================================================
# Model Configurations — Grid search space
# ============================================================================

from itertools import product

transformer_dims = [128, 192, 256]
transformer_heads_list = [4]
transformer_ff_list = [256, 384, 512]
transformer_layers_list = [3, 4, 6]
post_fc_options = [[64], [128, 64]]
dropouts = [0.2, 0.3]
activations = ['relu']
learning_rates = [5e-5]
epochs_list = [100]

model_configs = []
for t_dim, t_heads, t_ff, t_layers, post, dp, act in product(
        transformer_dims, transformer_heads_list, transformer_ff_list,
        transformer_layers_list, post_fc_options, dropouts, activations):
    # Skip invalid: heads must divide dim
    if t_dim % t_heads != 0:
        continue
    # Skip unreasonable combos: ff should be >= dim
    if t_ff < t_dim:
        continue
    cfg = {
        "waveform_length": 256,
        "cnn_embed_dim": 96,
        "pool_k": 8,
        "meta_dim": 12,  # PCA ON: freq(1) + last_level(1) + extracted(10)
        "output_size": num_classes,
        "pre_fc_layers": [],
        "transformer_dim": t_dim,
        "transformer_heads": t_heads,
        "transformer_ff": t_ff,
        "transformer_layers": t_layers,
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
    }
    model_configs.append(cfg)

print(f"\nGrid search: {len(model_configs)} configurations")


# ============================================================================
# Main: Grid Search / Train-Fixed / Eval
# ============================================================================

RESULTS_TAG = "non_auto_transformer_pca"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model (no training)")
    parser.add_argument("--train-fixed", action="store_true", help="Train with fixed config (no grid search)")
    parser.add_argument("--config", type=str, default=f"{ROOT_DIR}/exported_features/config/{RESULTS_TAG}_config.json")
    parser.add_argument("--weights", type=str, default=f"{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth")
    args = parser.parse_args()

    if args.eval:
        print(f"\n{'='*60}")
        print("EVAL MODE: Loading saved model and evaluating...")
        print('='*60)

        with open(args.config) as f:
            config = json.load(f)
        config["output_size"] = num_classes
        config["meta_dim"] = 12

        model = NonAutoTransformerSingleHead(config).to(DEVICE)
        model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=DEVICE))
        print(f"Loaded weights from: {args.weights}")
        print_model_architecture(model, config)

        accuracy, cm, results = testing_loop(model, test_split_prev_preds, reverse_mapping)

        print(f"\nTest Accuracy: {accuracy:.2f}%")
        plot_cm(cm, f'Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%',
               'Predicted Labels', 'True Labels', labels)

    elif args.train_fixed:
        print(f"\n{'='*60}")
        print(f"TRAIN-FIXED MODE: Training {RESULTS_TAG} with fixed config...")
        print('='*60)

        with open(args.config) as f:
            config = json.load(f)
        config["output_size"] = num_classes
        config["meta_dim"] = 12

        model = NonAutoTransformerSingleHead(config).to(DEVICE)
        print_model_architecture(model, config)

        accuracy, cm, results = run_model(
            model,
            epochs=100,
            lr=5e-5,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_set=test_split_prev_preds,
            label_map=mapping,
            reverse_map=reverse_mapping,
        )

        print(f"\nTest Accuracy: {accuracy:.2f}%")

        results_path = f"{ROOT_DIR}/exported_features/results/train/transformer/{RESULTS_TAG}_results.json"
        if is_higher_than_saved_acc(accuracy, results_path):
            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/transformer", exist_ok=True)
            torch.save(model.state_dict(), f'{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth')
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved new best model!")

        plot_cm(cm, f'Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%',
               'Predicted Labels', 'True Labels', labels)

    else:
        # Grid search mode
        print(f"\nExecuting {RESULTS_TAG} grid search...")

        best_acc = 0
        plt_best = None
        i = 1
        total = len(model_configs) * len(epochs_list) * len(learning_rates)

        random.shuffle(model_configs)

        for config in model_configs:
            for lr in learning_rates:
                for ep in epochs_list:
                    print(f"\n{'='*60}")
                    print(f"{i} / {total}: Epoch: {ep}, LR: {lr}")
                    print('='*60)

                    model = NonAutoTransformerSingleHead(config).to(DEVICE)
                    print_model_architecture(model, config)

                    accuracy, cm, results = run_model(
                        model,
                        epochs=ep,
                        lr=lr,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        test_set=test_split_prev_preds,
                        label_map=mapping,
                        reverse_map=reverse_mapping,
                    )

                    results_path = f"{ROOT_DIR}/exported_features/results/train/transformer/{RESULTS_TAG}_results.json"
                    if accuracy > best_acc and is_higher_than_saved_acc(accuracy, results_path):
                        print(f"\n*** Found better accuracy: {accuracy:.2f}%! ***")

                        os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
                        os.makedirs(f"{ROOT_DIR}/exported_features/config", exist_ok=True)
                        os.makedirs(f"{ROOT_DIR}/exported_features/results/train/transformer", exist_ok=True)

                        torch.save(model.state_dict(), f'{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth')
                        with open(f"{ROOT_DIR}/exported_features/config/{RESULTS_TAG}_config.json", "w") as f:
                            json.dump(config, f, indent=4)
                        with open(results_path, "w") as f:
                            json.dump(results, f, indent=4)

                        best_acc = accuracy
                        plt_best = cm
                        plot_cm(cm, f'Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%',
                               'Predicted Labels', 'True Labels', labels,
                               save_path=f"{ROOT_DIR}/exported_features/results/train/transformer/{RESULTS_TAG}_cm.png",
                               should_save=True)

                    i += 1

        print(f"\n{'='*60}")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print('='*60)

        if plt_best is not None:
            plot_cm(plt_best, f'Confusion Matrix ({RESULTS_TAG}) - {best_acc:.2f}%',
                   'Predicted Labels', 'True Labels', labels,
                   save_path=f"{ROOT_DIR}/exported_features/results/train/transformer/{RESULTS_TAG}_cm.png",
                   should_save=False)

