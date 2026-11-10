"""
Hybrid CNN-LSTM Implementation — With PCA + Decorrelation
==========================================================
Single-head model: predicts next action class directly (all classes in one head).

CNN waveform encoder + LSTM sequence model.
PCA ON → meta_dim = 12 (freq + last_level + 10 PCA features)
Decorrelation loss encourages CNN to learn features complementary to PCA.

Usage:
    cd src/models/impl
    python hybrid_cnn_lstm_impl.py
    python hybrid_cnn_lstm_impl.py --train-fixed --config f"{ROOT_DIR}/exported_features/config/best1.json"
    python hybrid_cnn_lstm_impl.py --eval
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
# Hybrid CNN-LSTM Model — Single Head, With PCA + Decorrelation
# ============================================================================

class HybridCNNLSTMModel(nn.Module):
    
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
        
        # Decorrelation projection: maps CNN embedding → PCA dim for correlation penalty
        pca_dim = config.get("pca_dim", 10)  # number of PCA features
        self.decorr_proj = nn.Linear(cnn_embed_dim, pca_dim, bias=False)
        
        # Embeddings
        embedding_dim = config.get("label_embedding_dim", 16)
        pos_embedding_dim = config.get("pos_embedding_dim", 16)
        output_dim = config["output_size"]
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.SiLU if activation == "silu" else nn.Tanh
        
        # Meta features: freq (1) + last_level (1) + PCA (10) = 12
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
        
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward_step(self, x_t, h, c):
        x_t = self.pre_fc(x_t)
        x_t = x_t.unsqueeze(0).unsqueeze(0)
        
        lstm_out, (h, c) = self.lstm(x_t, (h, c))
        hidden = self.post_fc(lstm_out)
        logits = self.output(hidden)
        return logits, h, c
    
    def forward_train(self, wave, meta, labels, sampling_prob=0.0):
        """
        wave: [T, 1, 256] - raw waveforms
        meta: [T, 12] - freq (1) + last_level (1) + PCA (10)
        labels: [T] - class labels
        
        Returns: (logits, labels, decorr_loss)
            decorr_loss: scalar penalizing CNN-PCA correlation
        """
        device = wave.device
        num_classes = self.output.out_features
        
        outputs, labels_out = [], []
        
        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        prev_class = num_classes  # Start token
        
        # Encode all waveforms at once
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_embeds = self.cnn_encoder(wave_norm)  # [T, cnn_dim]
        
        # Decorrelation loss: penalize CNN from learning what PCA already provides
        # Project CNN embeddings to PCA space, penalize cosine similarity
        pca_feats = meta[:, 2:]  # [T, 10] — PCA features only (skip freq + last_level)
        cnn_proj = self.decorr_proj(cnn_embeds)  # [T, 10]
        # Normalize both to unit vectors
        cnn_norm = nn.functional.normalize(cnn_proj, dim=-1)
        pca_norm = nn.functional.normalize(pca_feats, dim=-1)
        # Mean absolute cosine similarity across timesteps
        decorr_loss = (cnn_norm * pca_norm).sum(dim=-1).abs().mean()
        
        for t in range(len(wave)):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device, dtype=torch.long))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device, dtype=torch.long))
            
            # Concatenate: cnn_embed + meta + class_embed + pos_embed
            x_t = torch.cat([cnn_embeds[t], meta[t], prev_class_embed, pos_embed], dim=0)
            x_t = self.input_layer_norm(x_t)
            logits, h, c = self.forward_step(x_t, h, c)
            
            outputs.append(logits[0, 0])
            labels_out.append(labels[t])
            
            # Scheduled sampling
            if torch.rand(1).item() < sampling_prob:
                pred_class = torch.argmax(logits[0, 0]).item()
            else:
                pred_class = labels[t].item()
            
            prev_class = pred_class
        
        return torch.stack(outputs), torch.tensor(labels_out, device=device), decorr_loss
    
    def forward_test(self, wave, meta):
        """
        wave: [T, 1, 256]
        meta: [T, 12]
        """
        device = wave.device
        num_classes = self.output.out_features
        seq_len = wave.size(0)
        
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
            logits, h, c = self.forward_step(x_t, h, c)
            
            pred = torch.argmax(logits, dim=2)[0, 0].item()
            predictions.append(pred)
            
            prev_class = pred
            last_logits = logits[0, 0]
        
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

def ordinal_distance_loss(logits, labels, class_to_db, scale=0.03):
    """Auxiliary loss that penalizes predictions far from true dB value."""
    probs = torch.softmax(logits, dim=-1)
    db_valid = ~torch.isnan(class_to_db)
    db_values = class_to_db.clone()
    db_values[~db_valid] = 0
    valid_probs = probs * db_valid.float()
    prob_sum = valid_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized_probs = valid_probs / prob_sum
    expected_db = (normalized_probs * db_values).sum(dim=-1)
    true_db = class_to_db[labels]
    valid_mask = ~torch.isnan(true_db)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    huber = torch.nn.functional.smooth_l1_loss(expected_db[valid_mask], true_db[valid_mask], reduction='mean')
    invalid_mass = probs[valid_mask][:, ~db_valid].sum(dim=-1).mean()
    return scale * (huber + 0.1 * invalid_mass)

def compute_val_accuracy(model, val_prefixes):
    """Evaluate prefix-style validation and score only the last timestep."""
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

def is_higher_than_saved_acc(accuracy, saved_results_path):
    if not file_exists(saved_results_path):
        return True
    with open(saved_results_path) as f:
        results = json.load(f)
        prev_acc = results["accuracy"]
        return accuracy > prev_acc


# ============================================================================
# Training Loop
# ============================================================================

def training_loop(epochs, model, lr, train_dataset, val_prefixes, label_map, sampling_strategy, class_to_db):
    device = next(model.parameters()).device
    
    all_train_labels = []
    for _, _, y in train_dataset:
        all_train_labels.extend(y.cpu().numpy().tolist())
    
    class_weights = compute_class_weight('balanced', classes=np.array(list(label_map.values())), y=np.array(all_train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    class_to_db = class_to_db.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    steps_per_epoch = len(train_dataset)
    warmup_epochs = 20
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    print(f"  Val prefixes: {len(val_prefixes)} (1 forward pass/prefix, score last timestep)")
    
    patience = 25
    best_val_acc = 0.0
    best_loss = float('inf')
    best_model_state = None
    epochs_no_improve_loss = 0
    
    for epoch in range(epochs):
        total_loss = 0
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
            ord_loss = ordinal_distance_loss(output_flat, labels_flat, class_to_db, scale=0.03)
            loss = ce_loss + ord_loss + 0.05 * decorr_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        
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


def run_model(model, epochs, lr, sampling_strategy, train_dataset, val_prefixes, test_set, label_map, reverse_map, class_to_db):
    training_loop(
        epochs=epochs, model=model, lr=lr, 
        train_dataset=train_dataset, val_prefixes=val_prefixes,
        label_map=label_map, sampling_strategy=sampling_strategy,
        class_to_db=class_to_db
    )
    return testing_loop(model, test_set, reverse_map)


def print_model_architecture(model, config):
    print(f" CNN embed dim    : {config.get('cnn_embed_dim', 96)}")
    print(f" CNN pool_k       : {config.get('pool_k', 8)}")
    print(f" Meta dim         : {config.get('meta_dim', 12)} (freq+level+PCA)")
    print(f" Output size      : {config['output_size']}")
    print(f" LSTM hidden size : {config['lstm_hidden_size']}")
    print(f" LSTM layers      : {config['lstm_layers']}")
    print(f" Lbl Emb Dim      : {config['label_embedding_dim']}")
    print(f" Pos Emb Dim      : {config.get('pos_embedding_dim', 16)}")
    print(f" Dropout          : {config.get('dropout', 0.1)}")
    print(f" Activation       : {config.get('activation', 'relu')}")
    print(f" Pre-FC Layers    : {config.get('pre_fc_layers', [])}")
    print(f" Post-FC Layers   : {config.get('post_fc_layers', [])}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTOTAL PARAMETERS: {total_params:,}")


# ============================================================================
# Data Loading — PCA ON (include_pca=True, meta_dim=12)
# ============================================================================

import pandas as pd

print("Initializing data pipeline (Single-Head WITH PCA + Decorrelation)...")
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

# Create class -> dB mapping for ordinal loss
class_to_db = torch.full((num_classes,), float('nan'))
for raw_label, idx in mapping.items():
    if raw_label == "stop":
        continue
    if raw_label in (">20", "<-30"):
        continue
    try:
        class_to_db[idx] = float(raw_label)
    except ValueError:
        pass
print(f"  dB classes: {(~torch.isnan(class_to_db)).sum().item()} / {num_classes}")

# Load waveform data
print("Loading waveform data...")
waveform_df, record_to_bank = load_waveform_data()
print(f"  Loaded {len(waveform_df)} waveforms")

# Split: 80-10-10
print("Preparing sequences (CNN + PCA, include_pca=True)...")
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
    print(f"\n  Sample shapes (WITH PCA):")
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
learning_rates = [0.005]
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
        "meta_dim": 12,  # PCA ON: freq(1) + last_level(1) + PCA(10)
        "output_size": num_classes,
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

RESULTS_TAG = "hybrid_cnn_lstm"

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
        
        model = HybridCNNLSTMModel(config).to(DEVICE)
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
        
        model = HybridCNNLSTMModel(config).to(DEVICE)
        print_model_architecture(model, config)
        
        accuracy, cm, results = run_model(
            model, epochs=150, lr=0.0005, sampling_strategy="linear",
            train_dataset=train_dataset, val_prefixes=val_split_prev_preds,
            test_set=test_split_prev_preds, label_map=mapping,
            reverse_map=reverse_mapping, class_to_db=class_to_db
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
                        
                        model = HybridCNNLSTMModel(config).to(DEVICE)
                        print_model_architecture(model, config)
                        
                        accuracy, cm, results = run_model(
                            model, epochs=ep, lr=lr, sampling_strategy=sampling,
                            train_dataset=train_dataset, val_prefixes=val_split_prev_preds,
                            test_set=test_split_prev_preds, label_map=mapping,
                            reverse_map=reverse_mapping, class_to_db=class_to_db
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
