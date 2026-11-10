"""
Hybrid CNN-LSTM Two-Head Implementation — PCA variant
=====================================================
Two-headed model:
  - First head (3-class): STOP, REPEAT, ADJUST
  - Second head (multiclass): dB adjustments

Fixes over original twohead.py:
  (1) Device / dtype bugs in forward_test
  (2) Soft joint probability in forward_test (no hard argmax gating)
  (3) All-timestep loss (matches original twohead.py)
  (4) CNN encoder pool_k=8 for latency preservation, cnn_embed_dim=96
  (5) Criteria weights moved to model device

PCA ON  → meta_dim = 12 (freq + last_level + 10 extracted)

Usage:
    cd src/models/impl
    python hybrid_cnn_lstm_impl_twohead_pca.py
    python hybrid_cnn_lstm_impl_twohead_pca.py --train-fixed
    python hybrid_cnn_lstm_impl_twohead_pca.py --eval
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

# ============================================================================
# Hybrid CNN-LSTM Model with Stop+Repeat Head
# ============================================================================

class HybridCNNLSTMModelStopRepeat(nn.Module):
    
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
        
        # Embeddings
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
        
        # Pre-FC layers
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
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_layers"],
            dropout=dropout if config["lstm_layers"] > 1 else 0.0,
        )
        
        lstm_output_dim = config["lstm_hidden_size"]
        
        # Post-FC layers
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
        # Second head: dB adjustments (excludes stop and repeat)
        self.output_dim = output_dim
        self.stop_idx = config["stop_idx"]
        self.repeat_idx = config["repeat_idx"]
        self.action_classes = config["action_classes"]
        
        self.first_head = nn.Linear(prev_dim, 3)
        self.action_head = nn.Linear(prev_dim, self.action_classes)
    
    def forward_step(self, x_t, h, c):
        x_t = self.pre_fc(x_t)
        x_t = x_t.unsqueeze(0).unsqueeze(0)
        
        lstm_out, (h, c) = self.lstm(x_t, (h, c))
        hidden = self.post_fc(lstm_out)
        
        first_logits = self.first_head(hidden)
        action_logits = self.action_head(hidden)
        return first_logits, action_logits, h, c
    
    def forward_train(self, wave, meta, labels, sampling_prob=0.0, global_to_action=None, action_to_global=None):
        """
        wave: [T, 1, 256] - raw waveforms
        meta: [T, 12] - freq (1) + last_level (1) + extracted features (10)
        labels: [T] - class labels (global ids)
        """
        device = wave.device
        num_classes = self.output_dim
        stop_idx = self.stop_idx
        repeat_idx = self.repeat_idx
        
        first_logits_list = []
        action_logits_list = []
        first_targets_list = []
        action_targets_list = []
        
        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        prev_class = num_classes  # Start token
        
        # Encode all waveforms at once
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_embeds = self.cnn_encoder(wave_norm)
        
        for t in range(len(wave)):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device, dtype=torch.long))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device, dtype=torch.long))
            
            x_t = torch.cat([cnn_embeds[t], meta[t], prev_class_embed, pos_embed], dim=0)
            x_t = self.input_layer_norm(x_t)
            first_logits, action_logits, h, c = self.forward_step(x_t, h, c)
            
            first_logits_t = first_logits[0, 0]
            action_logits_t = action_logits[0, 0]
            
            true_gid = labels[t].item()
            
            if true_gid == stop_idx:
                first_target = FIRST_HEAD_STOP_IDX
            elif true_gid == repeat_idx:
                first_target = FIRST_HEAD_REPEAT_IDX
            else:
                first_target = FIRST_HEAD_ADJUST_IDX
            
            first_logits_list.append(first_logits_t)
            action_logits_list.append(action_logits_t)
            first_targets_list.append(first_target)
            
            if true_gid == stop_idx or true_gid == repeat_idx:
                action_targets_list.append(-100)
            else:
                action_targets_list.append(global_to_action[true_gid])
            
            # Scheduled sampling: use soft joint global_probs (matches forward_test)
            if torch.rand(1).item() < sampling_prob:
                first_probs = torch.softmax(first_logits_t, dim=0)
                action_probs = torch.softmax(action_logits_t, dim=0)
                global_probs = torch.zeros(num_classes, device=device)
                global_probs[stop_idx] = first_probs[FIRST_HEAD_STOP_IDX]
                global_probs[repeat_idx] = first_probs[FIRST_HEAD_REPEAT_IDX]
                for aid in range(action_probs.numel()):
                    gid = action_to_global[aid]
                    global_probs[gid] = first_probs[FIRST_HEAD_ADJUST_IDX] * action_probs[aid]
                prev_class = torch.argmax(global_probs).item()
            else:
                prev_class = true_gid
        
        return (
            torch.stack(first_logits_list),
            torch.stack(action_logits_list),
            torch.tensor(first_targets_list, device=device, dtype=torch.long),
            torch.tensor(action_targets_list, device=device, dtype=torch.long),
        )
    
    def forward_test(self, wave, meta, action_to_global=None):
        """
        FIX: device-aware, dtype-safe, soft joint probability (no hard argmax gating).
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
        cnn_embeds = self.cnn_encoder(wave_norm)
        
        for t in range(seq_len):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device, dtype=torch.long))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device, dtype=torch.long))
            
            x_t = torch.cat([cnn_embeds[t], meta[t], prev_class_embed, pos_embed], dim=0)
            x_t = self.input_layer_norm(x_t)
            first_logits, action_logits, h, c = self.forward_step(x_t, h, c)
            
            first_logits_t = first_logits[0, 0]      # [3]
            action_logits_t = action_logits[0, 0]    # [A]
            
            # --- Soft joint probability (no hard gating) ---
            first_probs = torch.softmax(first_logits_t, dim=0)      # [3]
            action_probs = torch.softmax(action_logits_t, dim=0)    # [A]
            
            global_probs = torch.zeros(num_classes, device=device)
            global_probs[stop_idx] = first_probs[FIRST_HEAD_STOP_IDX]
            global_probs[repeat_idx] = first_probs[FIRST_HEAD_REPEAT_IDX]
            
            for aid in range(action_probs.numel()):
                gid = action_to_global[aid]
                global_probs[gid] = first_probs[FIRST_HEAD_ADJUST_IDX] * action_probs[aid]
            
            pred_gid = torch.argmax(global_probs).item()
            predictions.append(pred_gid)
            prev_class = pred_gid
            last_logits = global_probs
        
        return torch.tensor(predictions, device=device), last_logits


# ============================================================================
# Helper functions
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
        if epoch < 10:
            return 0.0
        return min(0.3, (epoch - 10) / (epochs - 10) * 0.3)
    if strategy == "constant":
        return 0.3
    return 0.0

def compute_val_accuracy(model, val_prefixes, action_to_global):
    """Evaluate prefix-style validation and score only the last timestep."""
    model.eval()
    correct, total = 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for wave, meta, label in val_prefixes:
            wave = wave.to(device)
            meta = meta.to(device)
            label = label.to(device)
            pred, _ = model.forward_test(wave, meta, action_to_global)
            correct += int(pred[-1].item() == label[-1].item())
            total += 1
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
# Training Loop (Two-Head: First=STOP/REPEAT/ADJUST, Second=dB actions)
# ============================================================================

def training_loop(epochs, model, lr, train_dataset, val_prefixes, label_map, sampling_strategy, 
                  stop_idx, repeat_idx, action_classes, global_to_action, action_to_global):
    """Two-head training with all-timestep loss."""
    device = next(model.parameters()).device

    # --- First head class weights ---
    all_train_labels = []
    for _, _, y in train_dataset:
        all_train_labels.extend(y.cpu().numpy().tolist())
    train_labels_np = np.array(all_train_labels)
    
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
    
    # --- Action head class weights (handle classes missing from split) ---
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
    
    print(f"  Val prefixes: {len(val_prefixes)} (1 forward pass/prefix, score last timestep)")
    
    patience = 25
    best_val_acc = 0.0
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve_loss = 0
    first_batch_printed = False
    
    for epoch in range(epochs):
        total_loss = 0
        total_first, total_action = 0, 0
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
            
            first_logits, action_logits, first_tgt, action_tgt = model.forward_train(
                wave, meta, lbl, sampling_prob, global_to_action, action_to_global
            )
            
            first_loss = first_criterion(first_logits, first_tgt)
            
            # Action loss - handle case where all samples are STOP/REPEAT (all action_tgt = -100)
            valid_action_mask = (action_tgt != -100)
            if valid_action_mask.any():
                action_loss = action_criterion(action_logits, action_tgt)
                if torch.isnan(action_loss):
                    action_loss = torch.tensor(0.0, device=device)
            else:
                action_loss = torch.tensor(0.0, device=device)
            
            loss = first_loss + action_loss
            
            if not first_batch_printed:
                print(f"  [Debug] first_logits: {first_logits.shape}, action_logits: {action_logits.shape}")
                first_batch_printed = True
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_first += first_loss.item()
            total_action += action_loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_first = total_first / max(n_batches, 1)
        avg_action = total_action / max(n_batches, 1)
        
        val_acc = compute_val_accuracy(model, val_prefixes, action_to_global)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f} (First: {avg_first:.4f}, Action: {avg_action:.4f}), Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
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


def run_model(model, epochs, lr, sampling_strategy, train_dataset, val_prefixes, test_set, 
              label_map, reverse_map, stop_idx, repeat_idx, action_classes, global_to_action, action_to_global):
    training_loop(
        epochs=epochs, 
        model=model, 
        lr=lr, 
        train_dataset=train_dataset, 
        val_prefixes=val_prefixes,
        label_map=label_map, 
        sampling_strategy=sampling_strategy,
        stop_idx=stop_idx,
        repeat_idx=repeat_idx,
        action_classes=action_classes,
        global_to_action=global_to_action,
        action_to_global=action_to_global
    )
    return testing_loop(model, test_set, reverse_map, action_to_global)


def print_model_architecture(model, config):
    print(f" CNN embed dim    : {config.get('cnn_embed_dim', 96)}")
    print(f" CNN pool_k       : {config.get('pool_k', 8)}")
    print(f" Meta dim         : {config.get('meta_dim', 12)} (freq+level+extracted)")
    print(f" Output size      : {config['output_size']}")
    print(f" Action classes   : {config['action_classes']}")
    print(f" LSTM hidden size : {config['lstm_hidden_size']}")
    print(f" LSTM layers      : {config['lstm_layers']}")
    print(f" Lbl Emb Dim      : {config['label_embedding_dim']}")
    print(f" Pos Emb Dim      : {config.get('pos_embedding_dim', 16)}")
    print(f" Dropout          : {config.get('dropout', 0.1)}")
    print(f" Activation       : {config.get('activation', 'relu')}")
    print(f" Pre-FC Layers    : {config.get('pre_fc_layers', [])}")
    print(f" Post-FC Layers   : {config.get('post_fc_layers', [])}")
    print(f" First Head       : 3 classes (STOP, REPEAT, ADJUST)")
    print(f" Action Head      : {config['action_classes']} classes (dB adjustments)")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTOTAL PARAMETERS: {total_params:,}")


# ============================================================================
# Data Loading — HYBRID MODE: CNN + PCA (include_pca=True, meta_dim=12)
# ============================================================================

import pandas as pd

print("Initializing data pipeline (Two-Head PCA)...")
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/action_table.csv")
input_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/input.csv")

split_sessions = split_action_table(action_df)

action_df_clean = pd.concat(split_sessions, ignore_index=True)
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
# Model Configurations — cnn_embed_dim=96, pool_k=8, meta_dim=12
# ============================================================================

from itertools import product

hidden_sizes = [256, 320]
lstm_layers = [2, 3]
pre_fc_options = [[], [32]]
post_fc_options = [[64], [32], [128, 64]]
pos_embedding_dim_opts = [16, 24, 32]
label_embedding_dim_opts = [16, 24, 32]
dropouts = [0.2, 0.3]
activations = ['relu', 'silu']
learning_rates = [0.0005]
epochs_list = [200]
scheduled_sampling = ["linear"]

model_configs = []
for hs, nl, pre, post, dp, emb_dim, act, pos_emb_dim in product(
        hidden_sizes, lstm_layers, pre_fc_options, post_fc_options,
        dropouts, label_embedding_dim_opts, activations, pos_embedding_dim_opts):
    cfg = {
        "waveform_length": 256,
        "cnn_embed_dim": 96,
        "pool_k": 8,
        "meta_dim": 12,  # PCA ON: freq(1) + last_level(1) + extracted(10)
        "output_size": num_classes,
        "stop_idx": STOP_IDX,
        "repeat_idx": REPEAT_IDX,
        "action_classes": ACTION_CLASSES,
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
# Main: Grid Search / Train-Fixed / Eval
# ============================================================================

RESULTS_TAG = "hybrid_cnn_lstm_twohead_pca"

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
        config["meta_dim"] = 12  # Force PCA: freq + last_level + 10 extracted
        
        model = HybridCNNLSTMModelStopRepeat(config).to(DEVICE)
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
        config["meta_dim"] = 12  # Force PCA: freq + last_level + 10 extracted
        
        model = HybridCNNLSTMModelStopRepeat(config).to(DEVICE)
        print_model_architecture(model, config)
        
        accuracy, cm, results = run_model(
            model,
            epochs=150,
            lr=0.0005,
            sampling_strategy="linear",
            train_dataset=train_dataset,
            val_prefixes=val_split_prev_preds,
            test_set=test_split_prev_preds,
            label_map=mapping,
            reverse_map=reverse_mapping,
            stop_idx=STOP_IDX,
            repeat_idx=REPEAT_IDX,
            action_classes=ACTION_CLASSES,
            global_to_action=GLOBAL_TO_ACTION,
            action_to_global=ACTION_TO_GLOBAL
        )
        
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        results_path = f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_results.json"
        if is_higher_than_saved_acc(accuracy, results_path):
            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm", exist_ok=True)
            torch.save(model.state_dict(), f'{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth')
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved new best model!")
        
        plot_cm(cm, f'Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%', 
               'Predicted Labels', 'True Labels', labels)
    
    else:
        print(f"\nExecuting {RESULTS_TAG} grid search...")
        
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
                        
                        model = HybridCNNLSTMModelStopRepeat(config).to(DEVICE)
                        print_model_architecture(model, config)
                        
                        accuracy, cm, results = run_model(
                            model,
                            epochs=ep,
                            lr=lr,
                            sampling_strategy=sampling,
                            train_dataset=train_dataset,
                            val_prefixes=val_split_prev_preds,
                            test_set=test_split_prev_preds,
                            label_map=mapping,
                            reverse_map=reverse_mapping,
                            stop_idx=STOP_IDX,
                            repeat_idx=REPEAT_IDX,
                            action_classes=ACTION_CLASSES,
                            global_to_action=GLOBAL_TO_ACTION,
                            action_to_global=ACTION_TO_GLOBAL
                        )
                        
                        results_path = f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_results.json"
                        if accuracy > best_acc and is_higher_than_saved_acc(accuracy, results_path):
                            print(f"\n*** Found better accuracy: {accuracy:.2f}%! ***")
                            
                            os.makedirs(f"{ROOT_DIR}/exported_features/weights", exist_ok=True)
                            os.makedirs(f"{ROOT_DIR}/exported_features/config", exist_ok=True)
                            os.makedirs(f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm", exist_ok=True)
                            
                            torch.save(model.state_dict(), f'{ROOT_DIR}/exported_features/weights/{RESULTS_TAG}_weights.pth')
                            with open(f"{ROOT_DIR}/exported_features/config/{RESULTS_TAG}_config.json", "w") as f:
                                json.dump(config, f, indent=4)
                            with open(results_path, "w") as f:
                                json.dump(results, f, indent=4)
                            
                            best_acc = accuracy
                            plt_best = cm
                            plot_cm(cm, f'Confusion Matrix ({RESULTS_TAG}) - {accuracy:.2f}%', 
                                   'Predicted Labels', 'True Labels', labels, 
                                   save_path=f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_cm.png", 
                                   should_save=True)
                        
                        i += 1
        
        print(f"\n{'='*60}")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print('='*60)
    
        if plt_best is not None:
            plot_cm(plt_best, f'Confusion Matrix ({RESULTS_TAG}) - {best_acc:.2f}%', 
                   'Predicted Labels', 'True Labels', labels, 
                   save_path=f"{ROOT_DIR}/exported_features/results/train/hybrid_cnn_lstm/{RESULTS_TAG}_cm.png", 
                   should_save=False)
