"""
Hybrid CNN-LSTM Two-Head Implementation (Regression Head)
==========================================================
Two-headed model: 
  - First head (3-class): STOP, REPEAT, ADJUST (classification)
  - Second head (regression): Continuous dB delta prediction, snapped to nearest class at inference

This decouples the "should we stop/repeat?" decision from "what dB adjustment to make?".
The regression head learns to predict continuous dB values instead of discrete classes.

HYBRID MODE: CNN (pool_k=8, embed_dim=96) + PCA features (meta_dim=12)

Usage:
    cd src/models/impl
    python hybrid_cnn_lstm_impl_twohead_regression.py
    python hybrid_cnn_lstm_impl_twohead_regression.py --train-fixed
    python hybrid_cnn_lstm_impl_twohead_regression.py --eval
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

import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.waveform_loader import (
    load_waveform_data,
    WaveformSequenceDataset,
    split_for_testing_waveform
)
from src.common.model.waveform_cnn_encoder import ABRWaveformCNNEncoder

# ============================================================================
# First Head Classes: STOP, REPEAT, ADJUST
# ============================================================================
FIRST_HEAD_CLASSES = ["STOP", "REPEAT", "ADJUST"]
FIRST_HEAD_STOP_IDX = 0
FIRST_HEAD_REPEAT_IDX = 1
FIRST_HEAD_ADJUST_IDX = 2

# Copied from utils to avoid psycopg2/data_util dependency
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


def snap_to_nearest(value, allowed_values):
    """Snap a continuous value to the nearest allowed value."""
    allowed_arr = np.array(allowed_values)
    idx = np.abs(allowed_arr - value).argmin()
    return allowed_values[idx]


# ============================================================================
# Hybrid CNN-LSTM Model with Regression Head
# ============================================================================

class HybridCNNLSTMModelRegression(nn.Module):
    
    max_seq_len = 38
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # CNN encoder for waveforms (pool_k=8 to preserve latency info)
        cnn_embed_dim = config.get("cnn_embed_dim", 96)
        waveform_length = config.get("waveform_length", 256)
        dropout = config.get("dropout", 0.1)
        pool_k = config.get("pool_k", 8)
        
        self.cnn_encoder = ABRWaveformCNNEncoder(
            waveform_length=waveform_length,
            embed_dim=cnn_embed_dim,
            dropout=dropout,
            pool_k=pool_k
        )
        
        # Embeddings (same as ConfigurableLSTMModel)
        embedding_dim = config.get("label_embedding_dim", 16)
        pos_embedding_dim = config.get("pos_embedding_dim", 16)
        output_dim = config["output_size"]
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.SiLU if activation == "silu" else nn.Tanh
        
        # Meta features: freq (1) + last_level (1) + extracted features (10) = 12
        meta_dim = config.get("meta_dim", 12)

        # Input to pre-FC: cnn_embed + meta + class_embed + pos_embed
        input_dim = cnn_embed_dim + meta_dim + embedding_dim + pos_embedding_dim
        
        # LayerNorm on concatenated input
        self.input_layer_norm = nn.LayerNorm(input_dim)
        
        # Pre-FC layers (same as ConfigurableLSTMModel)
        pre_layers = []
        prev_dim = input_dim
        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)
        
        lstm_input_dim = prev_dim
        
        # Embeddings
        self.class_embedding = nn.Embedding(output_dim + 1, embedding_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)
        
        # LSTM (same as ConfigurableLSTMModel)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_layers"],
            dropout=dropout if config["lstm_layers"] > 1 else 0.0,
        )
        
        lstm_output_dim = config["lstm_hidden_size"]
        
        # Post-FC layers (same as ConfigurableLSTMModel)
        post_layers = []
        prev_dim = lstm_output_dim
        for h in config.get("post_fc_layers", []):
            post_layers.append(nn.Linear(prev_dim, h))
            post_layers.append(act_fn())
            post_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.post_fc = nn.Sequential(*post_layers)
        
        # Two-head architecture: 
        # First head: 3-class (STOP, REPEAT, ADJUST)
        # Second head: REGRESSION - continuous dB prediction
        self.output_dim = output_dim  # total classes (including stop and repeat)
        self.stop_idx = config["stop_idx"]
        self.repeat_idx = config["repeat_idx"]
        
        self.first_head = nn.Linear(prev_dim, 3)  # STOP, REPEAT, ADJUST
        self.delta_head = nn.Linear(prev_dim, 1)  # Regression: continuous dB output
    
    def forward_step(self, x_t, h, c):
        x_t = self.pre_fc(x_t)
        x_t = x_t.unsqueeze(0).unsqueeze(0)
        
        lstm_out, (h, c) = self.lstm(x_t, (h, c))
        hidden = self.post_fc(lstm_out)        # [1, 1, H]
        
        # Two heads
        first_logits = self.first_head(hidden)    # [1, 1, 3] - STOP, REPEAT, ADJUST
        delta_pred_raw = self.delta_head(hidden)  # [1, 1, 1] - raw continuous dB
        # Constrain delta to reasonable range [-35, +25] for training stability
        # This matches the extremes in ADJUST_DB_LIST (boundary values for <-30 and >20)
        delta_pred = torch.clamp(delta_pred_raw, min=-35.0, max=25.0)
        return first_logits, delta_pred, h, c
    
    def forward_train(self, wave, meta, labels, sampling_prob=0.0, gid_to_db=None, db_to_gid=None, adjust_db_list=None):
        """
        wave: [T, 1, 256] - raw waveforms
        meta: [T, 12] - freq (1) + last_level (1) + extracted features (10)
        labels: [T] - class labels (global ids)
        
        Returns:
            first_logits: [T, 3] - logits for STOP/REPEAT/ADJUST
            delta_preds: [T] - continuous dB predictions
            first_targets: [T] - 0=STOP, 1=REPEAT, 2=ADJUST
            delta_targets: [T] - float dB targets (0.0 for STOP/REPEAT, ignored via mask)
            delta_mask: [T] - bool mask (True only for ADJUST samples)
        """
        device = wave.device
        num_classes = self.output_dim
        stop_idx = self.stop_idx
        repeat_idx = self.repeat_idx
        
        first_logits_list = []
        delta_preds_list = []
        first_targets_list = []
        delta_targets_list = []
        delta_mask_list = []
        
        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        prev_class = num_classes  # Start token
        
        # Encode all waveforms at once
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_embeds = self.cnn_encoder(wave_norm)  # [T, cnn_dim]
        
        for t in range(len(wave)):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))
            
            # Concatenate: cnn_embed + meta + class_embed + pos_embed + LayerNorm
            x_t = torch.cat([cnn_embeds[t], meta[t], prev_class_embed, pos_embed], dim=0)
            x_t = self.input_layer_norm(x_t)
            first_logits, delta_pred, h, c = self.forward_step(x_t, h, c)
            
            # Flatten shapes
            first_logits_t = first_logits[0, 0]    # [3]
            delta_pred_t = delta_pred[0, 0, 0]     # scalar
            
            true_gid = labels[t].item()
            
            # Determine first head target and delta target
            if true_gid == stop_idx:
                first_target = FIRST_HEAD_STOP_IDX  # 0 = STOP
                delta_target = 0.0  # dummy, will be masked
                delta_mask = False
            elif true_gid == repeat_idx:
                first_target = FIRST_HEAD_REPEAT_IDX  # 1 = REPEAT
                delta_target = 0.0  # dummy, will be masked
                delta_mask = False
            else:
                first_target = FIRST_HEAD_ADJUST_IDX  # 2 = ADJUST
                delta_target = gid_to_db[true_gid]  # actual dB value
                delta_mask = True
            
            first_logits_list.append(first_logits_t)
            delta_preds_list.append(delta_pred_t)
            first_targets_list.append(first_target)
            delta_targets_list.append(delta_target)
            delta_mask_list.append(delta_mask)
            
            # Decide predicted global label for scheduled sampling
            first_pred = torch.argmax(first_logits_t).item()
            if first_pred == FIRST_HEAD_STOP_IDX:
                pred_gid = stop_idx
            elif first_pred == FIRST_HEAD_REPEAT_IDX:
                pred_gid = repeat_idx
            else:
                # Snap predicted delta to nearest allowed dB
                pred_delta = delta_pred_t.item()
                snapped_db = snap_to_nearest(pred_delta, adjust_db_list)
                pred_gid = db_to_gid[snapped_db]
            
            # Scheduled sampling
            if torch.rand(1).item() < sampling_prob:
                prev_class = pred_gid
            else:
                prev_class = true_gid
        
        return (
            torch.stack(first_logits_list),                              # [T, 3]
            torch.stack(delta_preds_list),                               # [T]
            torch.tensor(first_targets_list, device=device),             # [T] long
            torch.tensor(delta_targets_list, device=device, dtype=torch.float32),  # [T] float
            torch.tensor(delta_mask_list, device=device, dtype=torch.bool),        # [T] bool
        )
    
    def forward_test(self, wave, meta, gid_to_db=None, db_to_gid=None, adjust_db_list=None):
        """
        wave: [T, 1, 256]
        meta: [T, 12] - freq + last_level + extracted features
        
        Returns:
            predictions: [T] - global class ids
            last_logits: [C] - global probabilities for topk computation
        """
        device = wave.device
        num_classes = self.output_dim
        seq_len = wave.size(0)
        stop_idx = self.stop_idx
        repeat_idx = self.repeat_idx
        
        prev_class = num_classes
        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        
        predictions = []
        last_logits = None
        
        # Encode all waveforms
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_embeds = self.cnn_encoder(wave_norm)  # [T, cnn_dim]
        
        for t in range(seq_len):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device, dtype=torch.long))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device, dtype=torch.long))
            
            x_t = torch.cat([cnn_embeds[t], meta[t], prev_class_embed, pos_embed], dim=0)
            x_t = self.input_layer_norm(x_t)
            first_logits, delta_pred, h, c = self.forward_step(x_t, h, c)
            
            first_logits_t = first_logits[0, 0]    # [3]
            delta_pred_t = delta_pred[0, 0, 0].item()  # scalar
            
            first_pred = torch.argmax(first_logits_t).item()
            
            if first_pred == FIRST_HEAD_STOP_IDX:
                pred_gid = stop_idx
            elif first_pred == FIRST_HEAD_REPEAT_IDX:
                pred_gid = repeat_idx
            else:
                # Snap predicted delta to nearest allowed dB
                snapped_db = snap_to_nearest(delta_pred_t, adjust_db_list)
                pred_gid = db_to_gid[snapped_db]
            
            predictions.append(pred_gid)
            prev_class = pred_gid
            
            # Build global probabilities for topk
            # First head gives P(STOP), P(REPEAT), P(ADJUST)
            first_probs = torch.softmax(first_logits_t, dim=0)  # [3]
            
            # For ADJUST classes, create "action distribution" from regression
            # using distance-based scores: score_i = -|db_i - pred_delta|
            adjust_db_tensor = torch.tensor(adjust_db_list, dtype=torch.float32, device=device)
            distance_scores = -torch.abs(adjust_db_tensor - delta_pred_t)  # Higher is better
            action_probs = torch.softmax(distance_scores, dim=0)  # [num_adjust_classes]
            
            global_probs = torch.zeros(num_classes, device=device)
            global_probs[stop_idx] = first_probs[FIRST_HEAD_STOP_IDX]
            global_probs[repeat_idx] = first_probs[FIRST_HEAD_REPEAT_IDX]
            
            # For each adjust class: P(ADJUST) * P(that dB | ADJUST)
            for i, db in enumerate(adjust_db_list):
                gid = db_to_gid[db]
                global_probs[gid] = first_probs[FIRST_HEAD_ADJUST_IDX] * action_probs[i]
            
            last_logits = global_probs
        
        return torch.tensor(predictions, device=device), last_logits


# ============================================================================
# Helper functions (copied from model_util.py)
# ============================================================================

def split_action_table(action_df):
    split_indices = action_df.index[action_df['stop'] == 1].tolist()
    prev = 0
    dfs = []
    for index in split_indices:
        dfs.append(action_df.iloc[prev:index + 1, :])
        prev = index + 1
    return dfs

def get_sampling_probability(strategy, epoch, epochs):
    if strategy == "linear":
        # Warmup: no sampling first 10 epochs, then ramp to 0.3 max
        if epoch < 10:
            return 0.0
        return min(0.3, (epoch - 10) / (epochs - 10) * 0.3)
    if strategy == "constant":
        return 0.3
    return 0.0

def compute_val_accuracy(model, val_dataset, gid_to_db, db_to_gid, adjust_db_list):
    """1 forward pass per full sequence, score all T timesteps (causal = same as T prefix passes)."""
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for wave, meta, label in val_dataset:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)
            pred, _ = model.forward_test(wave, meta, gid_to_db, db_to_gid, adjust_db_list)
            correct += (pred == label).sum().item()
            total += label.size(0)
    model.train()
    return correct / total * 100 if total > 0 else 0

def is_higher_than_saved_acc(accuracy, saved_results_path):
    if not file_exists(saved_results_path):
        return True
    with open(saved_results_path) as f:
        results = json.load(f)
        prev_acc = results["accuracy"]
        return accuracy > prev_acc


# ============================================================================
# Training Loop (Two-Head: First=STOP/REPEAT/ADJUST, Second=Regression)
# ============================================================================

def training_loop(epochs, model, lr, train_dataset, val_dataset, label_map, sampling_strategy, 
                  stop_idx, repeat_idx, gid_to_db, db_to_gid, adjust_db_list):
    """Two-head training: CE for first head + Regression loss for delta head."""
    device = next(model.parameters()).device

    # Collect all training labels
    all_train_labels = []
    for _, _, y in train_dataset:
        all_train_labels.extend(y.cpu().numpy().tolist())
    train_labels_np = np.array(all_train_labels)
    
    # First head class weights
    first_head_labels = []
    for lbl in all_train_labels:
        if lbl == stop_idx:
            first_head_labels.append(FIRST_HEAD_STOP_IDX)
        elif lbl == repeat_idx:
            first_head_labels.append(FIRST_HEAD_REPEAT_IDX)
        else:
            first_head_labels.append(FIRST_HEAD_ADJUST_IDX)
    
    first_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=np.array(first_head_labels))
    first_weights = torch.tensor(first_weights, dtype=torch.float32).to(device)
    first_criterion = nn.CrossEntropyLoss(weight=first_weights)
    
    stop_count = np.sum(train_labels_np == stop_idx)
    repeat_count = np.sum(train_labels_np == repeat_idx)
    adjust_count = len(train_labels_np) - stop_count - repeat_count
    print(f"  [Two-head] First head distribution: STOP={stop_count}, REPEAT={repeat_count}, ADJUST={adjust_count}")
    
    # Regression loss: MSE penalizes large dB errors quadratically
    reg_criterion = nn.MSELoss(reduction="mean")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # Warmup for 20 epochs, then cosine decay for the rest
    steps_per_epoch = len(train_dataset)
    warmup_epochs = 20
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Create fixed validation prefixes
    print(f"  Val sequences: {len(val_dataset)} (1 forward pass/seq, all T timesteps)")
    
    patience = 25
    best_val_acc = 0.0
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve_loss = 0
    first_batch_printed = False
    
    for epoch in range(epochs):
        total_loss = 0
        total_first, total_reg = 0, 0
        n_batches = 0
        model.train()
        
        for wave, meta, lbl in train_dataset:
            # Move to device
            wave = wave.to(device)
            meta = meta.to(device)
            lbl = lbl.to(device)

            if torch.isnan(wave).any() or torch.isnan(meta).any():
                continue
            
            optimizer.zero_grad()
            sampling_prob = get_sampling_probability(sampling_strategy, epoch, epochs)
            
            first_logits, delta_preds, first_tgt, delta_tgt, delta_mask = model.forward_train(
                wave, meta, lbl, sampling_prob, gid_to_db, db_to_gid, adjust_db_list
            )
            
            first_loss = first_criterion(first_logits, first_tgt)
            
            # Regression loss - only on ADJUST samples (masked)
            if delta_mask.any():
                reg_loss = reg_criterion(delta_preds[delta_mask], delta_tgt[delta_mask])
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            loss = first_loss + reg_loss
            
            if not first_batch_printed:
                print(f"  [Debug] first_logits: {first_logits.shape}, delta_preds: {delta_preds.shape}")
                first_batch_printed = True
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_first += first_loss.item()
            total_reg += reg_loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_first = total_first / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)
        
        val_acc = compute_val_accuracy(model, val_dataset, gid_to_db, db_to_gid, adjust_db_list)
        
        current_lr = optimizer.param_groups[0]['lr']
        # Print both losses separate and combined
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f} (First: {avg_first:.4f}, Reg: {avg_reg:.4f}), Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
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

def testing_loop(model, test_set, reverse_map, gid_to_db, db_to_gid, adjust_db_list):
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
            
            pred, last_logits = model.forward_test(wave, meta, gid_to_db, db_to_gid, adjust_db_list)
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


def run_model(model, epochs, lr, sampling_strategy, train_dataset, val_dataset, test_set, 
              label_map, reverse_map, stop_idx, repeat_idx, gid_to_db, db_to_gid, adjust_db_list):
    training_loop(
        epochs=epochs, 
        model=model, 
        lr=lr, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset,
        label_map=label_map, 
        sampling_strategy=sampling_strategy,
        stop_idx=stop_idx,
        repeat_idx=repeat_idx,
        gid_to_db=gid_to_db,
        db_to_gid=db_to_gid,
        adjust_db_list=adjust_db_list
    )
    return testing_loop(model, test_set, reverse_map, gid_to_db, db_to_gid, adjust_db_list)


def print_model_architecture(model, config):
    print(f" CNN embed dim    : {config.get('cnn_embed_dim', 96)}")
    print(f" CNN pool_k       : {config.get('pool_k', 8)}")
    print(f" Meta dim         : {config.get('meta_dim', 12)} (freq+level+extracted)")
    print(f" Output size      : {config['output_size']}")
    print(f" LSTM hidden size : {config['lstm_hidden_size']}")
    print(f" LSTM layers      : {config['lstm_layers']}")
    print(f" Lbl Emb Dim      : {config['label_embedding_dim']}")
    print(f" Pos Emb Dim      : {config.get('pos_embedding_dim', 16)}")
    print(f" Dropout          : {config.get('dropout', 0.1)}")
    print(f" Activation       : {config.get('activation', 'relu')}")
    print(f" Pre-FC Layers    : {config.get('pre_fc_layers', [])}")
    print(f" Post-FC Layers   : {config.get('post_fc_layers', [])}")
    print(f" First Head       : 3 classes (STOP, REPEAT, ADJUST)")
    print(f" Delta Head       : Regression (continuous dB)")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTOTAL PARAMETERS: {total_params:,}")


# ============================================================================
# Data Loading (HYBRID MODE: CNN + PCA, meta_dim=12)
# ============================================================================

import pandas as pd

print("Initializing data pipeline (Regression Two-Head)...")
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/action_table.csv")
input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/input.csv")

split_sessions = split_action_table(action_df)

action_df_clean = pd.concat(split_sessions, ignore_index=True)
big_merged_df = pd.merge(input_df, action_df_clean, on="record_id")
label_ids = big_merged_df.iloc[:, 13:].idxmax(axis=1)
create_freq_graph(label_ids)

merged_dfs = [pd.merge(input_df, df, on='record_id') for df in split_sessions]
merged_dfs = [df for df in merged_dfs if len(df) > 0]

# Label mapping (same as hcnnt-new.py)
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

# Two-head model: First head = STOP/REPEAT/ADJUST, Second head = Regression
STOP_IDX = mapping["stop"]
REPEAT_IDX = mapping["0"]  # "0" maps to Repeat

# Build ADJUST-only dB mappings (exclude STOP, REPEAT only)
# Extremes get boundary values: >20 → 25, <-30 → -35
GID_TO_DB = {}  # global id -> dB value (for ADJUST classes only)
DB_TO_GID = {}  # dB value -> global id
ADJUST_DB_LIST = []  # sorted list of valid dB floats

for raw_label, gid in mapping.items():
    # Skip stop and repeat (0)
    if raw_label == "stop":
        continue
    if raw_label == "0":
        continue  # This is REPEAT, not an adjustment
    
    # Handle extremes with boundary values
    if raw_label == ">20":
        db_val = 25.0  # Boundary value for "too high"
    elif raw_label == "<-30":
        db_val = -35.0  # Boundary value for "too low"
    else:
        try:
            db_val = float(raw_label)
        except ValueError:
            continue
    
    GID_TO_DB[gid] = db_val
    DB_TO_GID[db_val] = gid
    ADJUST_DB_LIST.append(db_val)

# Sort the dB list for consistent snapping
ADJUST_DB_LIST = sorted(ADJUST_DB_LIST)

# CRITICAL ASSERTION: 0.0 must NOT be in ADJUST_DB_LIST (0 is REPEAT, not ADJUST)
assert 0.0 not in ADJUST_DB_LIST, "0.0 should not be in ADJUST_DB_LIST - it's handled by REPEAT!"

print(f"  Regression Two-head mode:")
print(f"    STOP_IDX={STOP_IDX}, REPEAT_IDX={REPEAT_IDX}")
print(f"    ADJUST_DB_LIST={ADJUST_DB_LIST}")
print(f"    Number of adjust classes: {len(ADJUST_DB_LIST)}")
print(f"    First head classes: {FIRST_HEAD_CLASSES}")

# Load waveform data
print("Loading waveform data...")
waveform_df, record_to_bank = load_waveform_data()
print(f"  Loaded {len(waveform_df)} waveforms")

# Split: 80-10-10 with shuffle
print("Preparing sequences (HYBRID mode: CNN + PCA)...")
print("Splitting into 80-10-10...")
random.shuffle(merged_dfs)
train_size = int(0.8 * len(merged_dfs))
val_size = int(0.1 * len(merged_dfs))

train_dfs = merged_dfs[:train_size]
val_dfs = merged_dfs[train_size:train_size + val_size]
test_dfs = merged_dfs[train_size + val_size:]

# include_pca=True for HYBRID mode (CNN + PCA)
# meta will be [T, 12]: freq(1) + last_level(1) + extracted(10)
train_dataset = WaveformSequenceDataset(train_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
val_dataset = WaveformSequenceDataset(val_dfs, mapping, waveform_df, record_to_bank, include_pca=True)
test_dataset = WaveformSequenceDataset(test_dfs, mapping, waveform_df, record_to_bank, include_pca=True)

print(f"  Train sequences: {len(train_dataset)} (80%)")
print(f"  Val sequences: {len(val_dataset)} (10%)")
print(f"  Test sequences: {len(test_dataset)} (10%)")

# Create test prefixes (same as split_indiv_records)
test_split_prev_preds = split_for_testing_waveform(test_dataset)
print(f"  Test prefixes: {len(test_split_prev_preds)}")

# Sanity check shapes
if len(train_dataset) > 0:
    wave, meta, y = train_dataset[0]
    print(f"\n  Sample shapes (HYBRID):")
    print(f"    wave: {wave.shape}")  # [T, 1, 256]
    print(f"    meta: {meta.shape}")  # [T, 12]
    print(f"    y: {y.shape}")        # [T]


# ============================================================================
# Model Configurations (same structure as lstm_final.py)
# ============================================================================

from itertools import product

# Grid search parameters (same as test_configurations.py)
hidden_sizes = [256, 320]
lstm_layers = [2, 3]
pre_fc_options = [[], [32]]
post_fc_options = [[64], [32], [128, 64]]
pos_embedding_dim = [16, 24, 32]
label_embedding_dim = [16, 24, 32]
dropouts = [0.2, 0.3]
activations = ['relu', 'silu']
learning_rates = [0.0005]
epochs_list = [200]
scheduled_sampling = ["linear"]

num_classes = len(mapping)

model_configs = []
for hs, nl, pre, post, dp, emb_dim, act, pos_emb_dim in product(
        hidden_sizes, lstm_layers, pre_fc_options, post_fc_options,
        dropouts, label_embedding_dim, activations, pos_embedding_dim):
    cfg = {
        "waveform_length": 256,
        "cnn_embed_dim": 96,
        "pool_k": 8,
        "meta_dim": 12,  # freq(1) + last_level(1) + extracted(10) = 12
        "output_size": num_classes,
        "stop_idx": STOP_IDX,
        "repeat_idx": REPEAT_IDX,
        "pre_fc_layers": pre,
        "lstm_hidden_size": hs,
        "lstm_layers": nl,
        "label_embedding_dim": emb_dim,
        "pos_embedding_dim": pos_emb_dim,
        "post_fc_layers": post,
        "dropout": dp,
        "activation": act,
    }
    model_configs.append(cfg)


# ============================================================================
# Grid Search (same as lstm_final.py)
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate saved model (no training)")
    parser.add_argument("--train-fixed", action="store_true", help="Train with fixed config (no grid search)")
    parser.add_argument("--config", type=str, default=f"{ROOT_DIR}/exported_features/config/hybrid_cnn_lstm_config.json")
    parser.add_argument("--weights", type=str, default=f"{ROOT_DIR}/exported_features/weights/hybrid_cnn_lstm_regression_weights.pth")
    args = parser.parse_args()
    
    if args.eval:
        # ========== EVAL MODE: Load weights and evaluate only ==========
        print(f"\n{'='*60}")
        print("EVAL MODE: Loading saved model and evaluating...")
        print('='*60)
        
        with open(args.config) as f:
            config = json.load(f)
        config["output_size"] = num_classes
        config["stop_idx"] = STOP_IDX
        config["repeat_idx"] = REPEAT_IDX
        
        model = HybridCNNLSTMModelRegression(config).to(DEVICE)
        model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=DEVICE))
        print(f"Loaded weights from: {args.weights}")
        print_model_architecture(model, config)
        
        accuracy, cm, results = testing_loop(model, test_split_prev_preds, reverse_mapping, GID_TO_DB, DB_TO_GID, ADJUST_DB_LIST)
        
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        plot_cm(cm, f'Confusion Matrix (Regression Two-Head) - {accuracy:.2f}%', 
               'Predicted Labels', 'True Labels', labels)
    
    elif args.train_fixed:
        # ========== TRAIN-FIXED MODE: Train with fixed config ==========
        print(f"\n{'='*60}")
        print("TRAIN-FIXED MODE: Training Regression Two-Head model with fixed config...")
        print('='*60)
        
        with open(args.config) as f:
            config = json.load(f)
        config["output_size"] = num_classes
        config["stop_idx"] = STOP_IDX
        config["repeat_idx"] = REPEAT_IDX
        
        model = HybridCNNLSTMModelRegression(config).to(DEVICE)
        print_model_architecture(model, config)
        
        accuracy, cm, results = run_model(
            model,
            epochs=150,
            lr=0.0005,
            sampling_strategy="linear",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_set=test_split_prev_preds,
            label_map=mapping,
            reverse_map=reverse_mapping,
            stop_idx=STOP_IDX,
            repeat_idx=REPEAT_IDX,
            gid_to_db=GID_TO_DB,
            db_to_gid=DB_TO_GID,
            adjust_db_list=ADJUST_DB_LIST
        )
        
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        # Save if better
        results_path = f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/hybrid_cnn_lstm_regression_results.json"
        if is_higher_than_saved_acc(accuracy, results_path):
            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm", exist_ok=True)
            torch.save(model.state_dict(), f'{ROOT_DIR}/exported_features/weights/hybrid_cnn_lstm_regression_weights.pth')
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved new best model!")
        
        plot_cm(cm, f'Confusion Matrix (Regression Two-Head) - {accuracy:.2f}%', 
               'Predicted Labels', 'True Labels', labels)
    
    else:
        # ========== GRID SEARCH MODE (default) ==========
        print(f"\nExecuting Regression Two-Head model tuning via grid search...")
        
        best_acc = 0
        plt_best = None
        i = 1
        total = len(model_configs) * len(epochs_list) * len(learning_rates) * len(scheduled_sampling)
        
        random.shuffle(model_configs)
        
        for config in model_configs:
            for lr in learning_rates:
                for ep in epochs_list:
                    for sampling in scheduled_sampling:
                        print(f"\n{'='*60}")
                        print(f"{i} / {total}: Epoch: {ep}, LR: {lr}, Sampling: {sampling}")
                        print('='*60)
                        
                        model = HybridCNNLSTMModelRegression(config).to(DEVICE)
                        print_model_architecture(model, config)
                        
                        accuracy, cm, results = run_model(
                            model,
                            epochs=ep,
                            lr=lr,
                            sampling_strategy=sampling,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            test_set=test_split_prev_preds,
                            label_map=mapping,
                            reverse_map=reverse_mapping,
                            stop_idx=STOP_IDX,
                            repeat_idx=REPEAT_IDX,
                            gid_to_db=GID_TO_DB,
                            db_to_gid=DB_TO_GID,
                            adjust_db_list=ADJUST_DB_LIST
                        )
                        
                        results_path = f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/hybrid_cnn_lstm_regression_results.json"
                        if accuracy > best_acc and is_higher_than_saved_acc(accuracy, results_path):
                            print(f"\n*** Found better accuracy: {accuracy:.2f}%! ***")
                            
                            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
                            os.makedirs(f"{ROOT_DIR}/exported_features/config", exist_ok=True)
                            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm", exist_ok=True)
                            
                            torch.save(model.state_dict(), f'{ROOT_DIR}/exported_features/weights/hybrid_cnn_lstm_regression_weights.pth')
                            with open(f"{ROOT_DIR}/exported_features/config/hybrid_cnn_lstm_regression_config.json", "w") as f:
                                json.dump(config, f, indent=4)
                            with open(results_path, "w") as f:
                                json.dump(results, f, indent=4)
                            
                            best_acc = accuracy
                            plt_best = cm
                            plot_cm(cm, f'Confusion Matrix (Regression Two-Head) - {accuracy:.2f}%', 
                                   'Predicted Labels', 'True Labels', labels, 
                                   save_path=f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/hybrid_cnn_lstm_regression_cm.png", 
                                   should_save=True)
                        
                        i += 1
        
        print(f"\n{'='*60}")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print('='*60)
    
        if plt_best is not None:
            plot_cm(plt_best, f'Confusion Matrix (Regression Two-Head) - {best_acc:.2f}%', 
                   'Predicted Labels', 'True Labels', labels, 
                   save_path=f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/hybrid_cnn_lstm_regression_cm.png", 
                   should_save=False)
