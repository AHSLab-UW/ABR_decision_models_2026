"""
Non-Autoregressive Transformer Single-Head Model
=================================================
Same CNN + Transformer architecture as the two-head variant, but with a single
output head that predicts all classes (STOP, REPEAT, dB adjustments) directly.

Simpler than two-head — no hierarchical decision. The model learns its own
internal structure for deciding between STOP/REPEAT/ADJUST.

Architecture:
  [T, 1, 256] waveforms  →  CNN encoder  →  [T, cnn_embed_dim]
       +  [T, meta_dim]  meta features  (freq, last_level, PCA)
       →  concat  →  LayerNorm  →  pre_fc  →  Linear proj  →  + pos_emb
       →  Transformer Encoder (causal mask)
       →  post_fc  →  output_head [T, num_classes]
"""

import torch
import torch.nn as nn
from src.common.model.waveform_cnn_encoder import ABRWaveformCNNEncoder


class NonAutoTransformerSingleHead(nn.Module):
    """
    Non-autoregressive transformer for ABR action prediction.

    Input per timestep: CNN(waveform) + meta (freq, last_level, PCA features)
    Output per timestep: single head predicting all classes directly

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

        # Input: CNN embedding + meta
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
        norm_first = config.get("norm_first", False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=config["transformer_heads"],
            dim_feedforward=config["transformer_ff"],
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
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

        # Single output head — all classes
        self.num_classes = config["output_size"]
        self.output_head = nn.Linear(prev_dim, self.num_classes)

    def _encode_sequence(self, wave, meta):
        """
        Encode entire sequence at once through CNN + Transformer.

        Args:
            wave: [T, 1, 256] raw waveforms
            meta: [T, meta_dim] non-waveform features

        Returns:
            logits: [T, num_classes]
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
        x = x.unsqueeze(0)  # [1, T, D]
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)  # [1, T, D]
        x = x.squeeze(0)  # [T, D]

        # Post-FC
        x = self.post_fc(x)

        # Single head
        logits = self.output_head(x)  # [T, num_classes]

        return logits

    def forward_train(self, wave, meta, labels):
        """
        Non-autoregressive training: predict all steps at once.

        Args:
            wave: [T, 1, 256] raw waveforms
            meta: [T, meta_dim] meta features
            labels: [T] global class indices

        Returns:
            logits: [T, num_classes]
            targets: [T] global class indices (same as labels)
        """
        logits = self._encode_sequence(wave, meta)
        return logits, labels

    def forward_test(self, wave, meta):
        """
        Non-autoregressive inference.

        Args:
            wave: [T, 1, 256] waveform prefix
            meta: [T, meta_dim] meta prefix

        Returns:
            predictions: [T] predicted global class indices
            last_logits: [num_classes] logits at final step
        """
        logits = self._encode_sequence(wave, meta)
        predictions = torch.argmax(logits, dim=-1)  # [T]
        last_logits = logits[-1]  # [num_classes]
        return predictions, last_logits
