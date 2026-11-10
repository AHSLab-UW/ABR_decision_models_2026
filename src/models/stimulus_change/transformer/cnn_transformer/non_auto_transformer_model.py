"""
Non-Autoregressive Transformer Two-Head Model
==============================================
Processes entire sequence at once — no feeding previous predictions.
Causal mask ensures each step only attends to current and previous steps.

Two-head output:
  - First head: STOP / REPEAT / ADJUST (3-class)
  - Action head: dB adjustments (excludes stop and repeat)

Architecture:
  [T, 1, 256] waveforms  →  CNN encoder  →  [T, cnn_embed_dim]
       +  [T, meta_dim]  meta features  (freq, last_level, PCA)
       →  concat  →  LayerNorm  →  pre_fc  →  Linear proj  →  + pos_emb
       →  Transformer Encoder (causal mask)
       →  post_fc  →  first_head [T, 3]  +  action_head [T, A]
"""

import torch
import torch.nn as nn
from src.common.model.waveform_cnn_encoder import ABRWaveformCNNEncoder

# First head class indices (must match impl)
FIRST_HEAD_STOP_IDX = 0
FIRST_HEAD_REPEAT_IDX = 1
FIRST_HEAD_ADJUST_IDX = 2


class NonAutoTransformerTwoHead(nn.Module):
    """
    Non-autoregressive transformer for ABR action prediction.

    Input per timestep: CNN(waveform) + meta (freq, last_level, PCA features)
    Output per timestep: first_head (3-class) + action_head (dB adjustments)

    No previous action feedback — uses causal masking for temporal ordering.
    """

    max_seq_len = 64  # Must be >= longest sequence

    def __init__(self, config):
        super().__init__()
        self.config = config

        # CNN encoder for waveforms
        cnn_embed_dim = config.get("cnn_embed_dim", 96)
        waveform_length = config.get("waveform_length", 256)
        dropout = config.get("dropout", 0.2)
        pool_k = config.get("pool_k", 8)

        self.cnn_encoder = ABRWaveformCNNEncoder(
            waveform_length=waveform_length,
            embed_dim=cnn_embed_dim,
            dropout=dropout,
            pool_k=pool_k
        )

        # Meta features: freq (1) + last_level (1) + extracted (10) = 12
        meta_dim = config.get("meta_dim", 12)

        # Input: CNN embedding + meta (NO previous action embedding)
        input_dim = cnn_embed_dim + meta_dim

        # Input LayerNorm for stability
        self.input_layer_norm = nn.LayerNorm(input_dim)

        # Activation function
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.SiLU if activation == "silu" else nn.Tanh

        # Pre-transformer FC layers (optional dimensionality adjustment)
        pre_layers = []
        prev_dim = input_dim
        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        # Project to transformer dimension
        transformer_dim = config["transformer_dim"]
        self.input_proj = nn.Linear(prev_dim, transformer_dim)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(self.max_seq_len, transformer_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=config["transformer_heads"],
            dim_feedforward=config["transformer_ff"],
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["transformer_layers"],
        )

        # Post-transformer FC layers (optional)
        post_layers = []
        prev_dim = transformer_dim
        for h in config.get("post_fc_layers", []):
            post_layers.append(nn.Linear(prev_dim, h))
            post_layers.append(act_fn())
            post_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.post_fc = nn.Sequential(*post_layers)

        # Two-head output
        self.output_dim = config["output_size"]
        self.stop_idx = config["stop_idx"]
        self.repeat_idx = config["repeat_idx"]
        self.action_classes = config["action_classes"]

        self.first_head = nn.Linear(prev_dim, 3)       # STOP, REPEAT, ADJUST
        self.action_head = nn.Linear(prev_dim, self.action_classes)  # dB adjustments

    def _encode_sequence(self, wave, meta):
        """
        Encode entire sequence at once through CNN + Transformer.

        Args:
            wave: [T, 1, 256] raw waveforms
            meta: [T, meta_dim] non-waveform features

        Returns:
            first_logits: [T, 3]
            action_logits: [T, action_classes]
        """
        T = wave.size(0)
        device = wave.device

        # CNN encode all waveforms at once
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_emb = self.cnn_encoder(wave_norm)  # [T, cnn_dim]

        # Concatenate CNN + meta
        x = torch.cat([cnn_emb, meta], dim=-1)  # [T, cnn_dim + meta_dim]
        x = self.input_layer_norm(x)

        # Pre-FC
        x = self.pre_fc(x)

        # Project to transformer dim
        x = self.input_proj(x)  # [T, transformer_dim]

        # Add positional encoding
        pos = torch.arange(T, device=device)
        x = x + self.pos_emb(pos)

        # Transformer with causal mask
        # batch_first=True: expects (batch, seq_len, d_model)
        x = x.unsqueeze(0)  # [1, T, D]
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)  # [1, T, D]
        x = x.squeeze(0)  # [T, D]

        # Post-FC
        x = self.post_fc(x)

        # Two heads
        first_logits = self.first_head(x)    # [T, 3]
        action_logits = self.action_head(x)  # [T, A]

        return first_logits, action_logits

    def forward_train(self, wave, meta, labels, global_to_action=None):
        """
        Non-autoregressive training: predict all steps at once.
        No scheduled sampling needed.

        Args:
            wave: [T, 1, 256] raw waveforms
            meta: [T, meta_dim] meta features
            labels: [T] global class indices

        Returns:
            first_logits: [T, 3]
            action_logits: [T, action_classes]
            first_targets: [T] (0=STOP, 1=REPEAT, 2=ADJUST)
            action_targets: [T] (-100 for STOP/REPEAT, else action class)
        """
        first_logits, action_logits = self._encode_sequence(wave, meta)

        # Build targets
        device = wave.device
        first_targets = []
        action_targets = []

        for t in range(len(labels)):
            gid = labels[t].item()
            if gid == self.stop_idx:
                first_targets.append(FIRST_HEAD_STOP_IDX)
            elif gid == self.repeat_idx:
                first_targets.append(FIRST_HEAD_REPEAT_IDX)
            else:
                first_targets.append(FIRST_HEAD_ADJUST_IDX)

            if gid == self.stop_idx or gid == self.repeat_idx:
                action_targets.append(-100)
            else:
                action_targets.append(global_to_action[gid])

        return (
            first_logits,
            action_logits,
            torch.tensor(first_targets, device=device, dtype=torch.long),
            torch.tensor(action_targets, device=device, dtype=torch.long),
        )

    def forward_test(self, wave, meta, action_to_global=None):
        """
        Non-autoregressive inference.
        Feed prefix, get predictions at ALL positions in one shot.

        Args:
            wave: [T, 1, 256] waveform prefix
            meta: [T, meta_dim] meta prefix

        Returns:
            predictions: [T] predicted global class indices
            last_logits: [num_classes] joint probability at final step
        """
        device = wave.device
        first_logits, action_logits = self._encode_sequence(wave, meta)

        T = first_logits.size(0)
        predictions = []
        last_logits = None

        for t in range(T):
            first_probs = torch.softmax(first_logits[t], dim=0)   # [3]
            action_probs = torch.softmax(action_logits[t], dim=0)  # [A]

            # Soft joint probability (same as LSTM two-head)
            global_probs = torch.zeros(self.output_dim, device=device)
            global_probs[self.stop_idx] = first_probs[FIRST_HEAD_STOP_IDX]
            global_probs[self.repeat_idx] = first_probs[FIRST_HEAD_REPEAT_IDX]

            for aid in range(action_probs.numel()):
                gid = action_to_global[aid]
                global_probs[gid] = first_probs[FIRST_HEAD_ADJUST_IDX] * action_probs[aid]

            pred_gid = torch.argmax(global_probs).item()
            predictions.append(pred_gid)
            last_logits = global_probs

        return torch.tensor(predictions, device=device), last_logits

