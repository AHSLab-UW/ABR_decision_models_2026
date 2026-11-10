"""
Non-Autoregressive Transformer Two-Head — PCA variant
=====================================================
NON-AUTOREGRESSIVE: no previous-action feedback, no compounding errors.
The model sees CNN(waveform) + meta (freq, last_level, PCA) at each step,
uses causal masking so step t only sees steps 0..t, and predicts every action
in a single forward pass.

Two-headed model:
  - First head (3-class): STOP, REPEAT, ADJUST
  - Second head (multiclass): dB adjustments

PCA ON → meta_dim = 12 (freq + last_level + 10 extracted)

Usage:
    cd src/models/impl
    python non_auto_transformer_twohead_pca_impl.py
    python non_auto_transformer_twohead_pca_impl.py --train-fixed
    python non_auto_transformer_twohead_pca_impl.py --eval
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))

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

import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.waveform_loader import (
    load_waveform_data,
    WaveformSequenceDataset,
    split_for_testing_waveform
)
from src.models.stimulus_change.transformer.cnn_transformer.non_auto_transformer_model import NonAutoTransformerTwoHead

# ============================================================================
# First Head Classes: STOP, REPEAT, ADJUST
# ============================================================================
FIRST_HEAD_CLASSES = ["STOP", "REPEAT", "ADJUST"]

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
# Validation — prefix-style evaluation, score final timestep of each prefix
# ============================================================================

def compute_val_accuracy(model, val_prefixes, action_to_global):
    """
    Validation accuracy using the same prefix-style evaluation as test time.
    Each prefix is fed as its own sequence and only the final timestep is scored.
    """
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


# ============================================================================
# Ordinal Distance Loss — for action head (improves top-3 accuracy)
# ============================================================================

def ordinal_loss_action_head(action_logits, action_targets, action_class_to_db, scale=0.1):
    """
    Auxiliary loss that penalizes dB predictions far from true dB value.
    Encourages probability mass on nearby dB classes → improves top-3 accuracy.

    Only applies to ADJUST steps (action_targets != -100).
    Uses Huber loss for robustness to outliers (<-30, >20 mapped as NaN).
    """
    device = action_logits.device
    valid_mask = (action_targets != -100)
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    valid_logits = action_logits[valid_mask]    # [N, A]
    valid_targets = action_targets[valid_mask]   # [N]

    probs = torch.softmax(valid_logits, dim=-1)  # [N, A]

    acdb = action_class_to_db.to(device)
    db_valid = ~torch.isnan(acdb)   # [A]
    db_values = acdb.clone()
    db_values[~db_valid] = 0

    # Expected dB = probability-weighted average over valid dB classes
    valid_probs = probs * db_valid.float()
    prob_sum = valid_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized_probs = valid_probs / prob_sum
    expected_db = (normalized_probs * db_values).sum(dim=-1)  # [N]

    # True dB for each sample
    true_db = acdb[valid_targets]   # [N]
    ordinal_valid = ~torch.isnan(true_db)

    if not ordinal_valid.any():
        return torch.tensor(0.0, device=device)

    huber = torch.nn.functional.smooth_l1_loss(
        expected_db[ordinal_valid],
        true_db[ordinal_valid],
        reduction='mean'
    )

    return scale * huber


# ============================================================================
# Training Loop (Non-Autoregressive — no scheduled sampling)
# ============================================================================

def training_loop(epochs, model, lr, train_dataset, val_prefixes, label_map,
                  stop_idx, repeat_idx, action_classes, global_to_action, action_to_global,
                  action_class_to_db=None):
    """Non-autoregressive two-head training with all-timestep loss, ordinal loss, and gradient accumulation."""
    device = next(model.parameters()).device

    # --- First head class weights ---
    all_train_labels = []
    for _, _, y in train_dataset:
        all_train_labels.extend(y.cpu().numpy().tolist())
    train_labels_np = np.array(all_train_labels)

    first_head_labels = []
    for lbl in all_train_labels:
        if lbl == stop_idx:
            first_head_labels.append(0)  # STOP
        elif lbl == repeat_idx:
            first_head_labels.append(1)  # REPEAT
        else:
            first_head_labels.append(2)  # ADJUST

    first_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=np.array(first_head_labels))
    first_weights = torch.tensor(first_weights, dtype=torch.float32).to(device)
    first_criterion = nn.CrossEntropyLoss(weight=first_weights)

    stop_count = np.sum(train_labels_np == stop_idx)
    repeat_count = np.sum(train_labels_np == repeat_idx)
    adjust_count = len(train_labels_np) - stop_count - repeat_count
    print(f"  [Two-head] First head distribution: STOP={stop_count}, REPEAT={repeat_count}, ADJUST={adjust_count}")

    # --- Action head class weights ---
    action_train = [global_to_action[y] for y in all_train_labels if y != stop_idx and y != repeat_idx]
    action_train_np = np.array(action_train)
    action_classes_arr = np.arange(action_classes)
    present = np.unique(action_train_np)
    pw = compute_class_weight('balanced', classes=present, y=action_train_np)
    wmap = dict(zip(present, pw))
    fallback = float(np.median(pw))
    action_weights = np.array([wmap.get(c, fallback) for c in action_classes_arr])
    action_weights = torch.tensor(action_weights, dtype=torch.float32).to(device)
    action_criterion = nn.CrossEntropyLoss(weight=action_weights, ignore_index=-100)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3,
                            betas=(0.9, 0.98), eps=1e-9)

    # Warmup for 10 epochs, then cosine decay
    # (Transformers are more LR-sensitive than LSTMs — shorter warmup, lower peak)
    steps_per_epoch = len(train_dataset)
    warmup_epochs = 5
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

    print(f"  Val prefixes: {len(val_prefixes)} (prefix-style validation, final timestep scored)")

    patience = 25
    best_val_acc = 0.0
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve_loss = 0
    first_batch_printed = False

    for epoch in range(epochs):
        total_loss = 0
        total_first, total_action, total_ord = 0, 0, 0
        n_batches = 0
        model.train()

        for seq_idx, (wave, meta, lbl) in enumerate(train_dataset):
            wave = wave.to(device)
            meta = meta.to(device)
            lbl = lbl.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue

            # Non-autoregressive: predict ALL timesteps at once
            first_logits, action_logits, first_tgt, action_tgt = model.forward_train(
                wave, meta, lbl, global_to_action
            )

            first_loss = first_criterion(first_logits, first_tgt)

            # Action loss — handle case where all samples are STOP/REPEAT
            valid_action_mask = (action_tgt != -100)
            if valid_action_mask.any():
                action_loss = action_criterion(action_logits, action_tgt)
                if torch.isnan(action_loss):
                    action_loss = torch.tensor(0.0, device=device)
            else:
                action_loss = torch.tensor(0.0, device=device)

            # Ordinal distance loss for action head (improves top-3 accuracy)
            if action_class_to_db is not None:
                ord_loss = ordinal_loss_action_head(action_logits, action_tgt, action_class_to_db)
            else:
                ord_loss = torch.tensor(0.0, device=device)

            loss = first_loss + action_loss + ord_loss

            if not first_batch_printed:
                print(f"  [Debug] first_logits: {first_logits.shape}, action_logits: {action_logits.shape}")
                print(f"  [Debug] Ordinal loss: {'ENABLED' if action_class_to_db is not None else 'DISABLED'}")
                first_batch_printed = True

            loss.backward()

            total_loss += loss.item()
            total_first += first_loss.item()
            total_action += action_loss.item()
            total_ord += ord_loss.item()
            n_batches += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        avg_first = total_first / max(n_batches, 1)
        avg_action = total_action / max(n_batches, 1)
        avg_ord = total_ord / max(n_batches, 1)

        val_acc = compute_val_accuracy(model, val_prefixes, action_to_global)

        current_lr = optimizer.param_groups[0]['lr']
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


# ============================================================================
# Testing Loop
# ============================================================================

def testing_loop(model, test_set, reverse_map, action_to_global):
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

            pred, last_logits = model.forward_test(wave, meta, action_to_global)
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


def run_model(model, epochs, lr, train_dataset, val_prefixes, test_set,
              label_map, reverse_map, stop_idx, repeat_idx, action_classes,
              global_to_action, action_to_global, action_class_to_db=None):
    training_loop(
        epochs=epochs,
        model=model,
        lr=lr,
        train_dataset=train_dataset,
        val_prefixes=val_prefixes,
        label_map=label_map,
        stop_idx=stop_idx,
        repeat_idx=repeat_idx,
        action_classes=action_classes,
        global_to_action=global_to_action,
        action_to_global=action_to_global,
        action_class_to_db=action_class_to_db,
    )
    return testing_loop(model, test_set, reverse_map, action_to_global)


def print_model_architecture(model, config):
    print(f" CNN embed dim         : {config.get('cnn_embed_dim', 96)}")
    print(f" CNN pool_k            : {config.get('pool_k', 8)}")
    print(f" Meta dim              : {config.get('meta_dim', 12)} (freq+level+extracted)")
    print(f" Output size           : {config['output_size']}")
    print(f" Action classes        : {config['action_classes']}")
    print(f" Transformer dim       : {config['transformer_dim']}")
    print(f" Transformer heads     : {config['transformer_heads']}")
    print(f" Transformer FF        : {config['transformer_ff']}")
    print(f" Transformer layers    : {config['transformer_layers']}")
    print(f" Dropout               : {config.get('dropout', 0.15)}")
    print(f" Activation            : {config.get('activation', 'relu')}")
    print(f" Pre-FC Layers         : {config.get('pre_fc_layers', [])}")
    print(f" Post-FC Layers        : {config.get('post_fc_layers', [])}")
    print(f" First Head            : 3 classes (STOP, REPEAT, ADJUST)")
    print(f" Action Head           : {config['action_classes']} classes (dB adjustments)")
    print(f" Mode                  : NON-AUTOREGRESSIVE (no prev action feedback)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTOTAL PARAMETERS: {total_params:,}")


# ============================================================================
# Data Loading — HYBRID MODE: CNN + PCA (include_pca=True, meta_dim=12)
# ============================================================================

import pandas as pd

print("Initializing data pipeline (Non-Auto Transformer Two-Head PCA — ALL classes)...")
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/action_table.csv")
input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/input.csv")

# NO rare class dropping — keep everything
split_sessions = split_action_table(action_df)
print(f"  Total sequences: {len(split_sessions)} (ALL classes kept, nothing dropped)")

big_merged_df = pd.merge(input_df, action_df, on="record_id")
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
print(f"Label Mapping Calculated!: {sorted(reverse_mapping.keys())}")

# Two-head indices
STOP_IDX = mapping["stop"]
REPEAT_IDX = mapping["0"]
ACTION_CLASSES = num_classes - 2

GLOBAL_TO_ACTION = {}
ACTION_TO_GLOBAL = {}
aid = 0
for gid in range(num_classes):
    if gid == STOP_IDX or gid == REPEAT_IDX:
        continue
    GLOBAL_TO_ACTION[gid] = aid
    ACTION_TO_GLOBAL[aid] = gid
    aid += 1

print(f"  Two-head mode:")
print(f"    STOP_IDX={STOP_IDX}, REPEAT_IDX={REPEAT_IDX}")
print(f"    ACTION_CLASSES={ACTION_CLASSES}")
print(f"    First head classes: {FIRST_HEAD_CLASSES}")

# Build action_class_to_db tensor for ordinal loss
index_to_colname = {v: k for k, v in mapping.items()}
action_db_values = []
for aid in range(ACTION_CLASSES):
    gid = ACTION_TO_GLOBAL[aid]
    colname = index_to_colname[gid]
    if colname.startswith('<') or colname.startswith('>'):
        action_db_values.append(float('nan'))  # overflow classes — excluded from ordinal loss
    else:
        action_db_values.append(float(colname))
ACTION_CLASS_TO_DB = torch.tensor(action_db_values, dtype=torch.float32)
print(f"    Action dB mapping: {action_db_values}")

# Load waveform data
print("Loading waveform data...")
waveform_df, record_to_bank = load_waveform_data()
print(f"  Loaded {len(waveform_df)} waveforms")

# Split: 80-10-10
print("Preparing sequences (HYBRID mode: CNN + PCA, include_pca=True)...")
print("Splitting into 80-10-10...")
random.shuffle(merged_dfs)
train_size = int(0.80 * len(merged_dfs))
val_size = int(0.10 * len(merged_dfs))

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

val_split_prev_preds = split_for_testing_waveform(val_dataset)
test_split_prev_preds = split_for_testing_waveform(test_dataset)
print(f"  Val prefixes: {len(val_split_prev_preds)}")
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
        "stop_idx": STOP_IDX,
        "repeat_idx": REPEAT_IDX,
        "action_classes": ACTION_CLASSES,
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

RESULTS_TAG = "non_auto_transformer_twohead_pca"

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
        config["stop_idx"] = STOP_IDX
        config["repeat_idx"] = REPEAT_IDX
        config["action_classes"] = ACTION_CLASSES
        config["meta_dim"] = 12

        model = NonAutoTransformerTwoHead(config).to(DEVICE)
        model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=DEVICE))
        print(f"Loaded weights from: {args.weights}")
        print_model_architecture(model, config)

        accuracy, cm, results = testing_loop(model, test_split_prev_preds, reverse_mapping, ACTION_TO_GLOBAL)

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
        config["stop_idx"] = STOP_IDX
        config["repeat_idx"] = REPEAT_IDX
        config["action_classes"] = ACTION_CLASSES
        config["meta_dim"] = 12

        model = NonAutoTransformerTwoHead(config).to(DEVICE)
        print_model_architecture(model, config)

        accuracy, cm, results = run_model(
            model,
            epochs=150,
            lr=5e-5,
            train_dataset=train_dataset,
            val_prefixes=val_split_prev_preds,
            test_set=test_split_prev_preds,
            label_map=mapping,
            reverse_map=reverse_mapping,
            stop_idx=STOP_IDX,
            repeat_idx=REPEAT_IDX,
            action_classes=ACTION_CLASSES,
            global_to_action=GLOBAL_TO_ACTION,
            action_to_global=ACTION_TO_GLOBAL,
            action_class_to_db=ACTION_CLASS_TO_DB,
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

                    model = NonAutoTransformerTwoHead(config).to(DEVICE)
                    print_model_architecture(model, config)

                    accuracy, cm, results = run_model(
                        model,
                        epochs=ep,
                        lr=lr,
                        train_dataset=train_dataset,
                        val_prefixes=val_split_prev_preds,
                        test_set=test_split_prev_preds,
                        label_map=mapping,
                        reverse_map=reverse_mapping,
                        stop_idx=STOP_IDX,
                        repeat_idx=REPEAT_IDX,
                        action_classes=ACTION_CLASSES,
                        global_to_action=GLOBAL_TO_ACTION,
                        action_to_global=ACTION_TO_GLOBAL,
                        action_class_to_db=ACTION_CLASS_TO_DB,
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
