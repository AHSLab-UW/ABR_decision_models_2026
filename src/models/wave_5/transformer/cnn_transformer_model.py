import torch
import torch.nn as nn

from src.common.model.api.base_model import ConfigurableModelBase
from src.common.model.waveform_cnn_encoder import ABRWaveformCNNEncoder


class Wave5CNNTransformerModel(ConfigurableModelBase):
    """
    CNN + Causal Transformer for binary Wave V classification.

    Replaces the PCA feature extraction with ABRWaveformCNNEncoder — a
    learnable 1D CNN that uses GroupNorm (stable for small session batches),
    dilated convolutions (multi-scale morphology), and a LayerNorm'd output
    embedding. The CNN output is concatenated with metadata (freq + dB) and
    positional embeddings before being fed into a causal Transformer encoder.

    Input tensor layout (per timestep):
        [freq, last_level, value_0, value_1, ..., value_255]  → 258 dims total
        Columns 0:2  = metadata (freq, dB)
        Columns 2:   = raw 256-point waveform

    Architecture:
        Raw waveform (256) → ABRWaveformCNNEncoder → embedding (cnn_output_dim)
        [CNN embedding + freq + dB + positional embedding]
        → Pre-FC layers
        → Linear projection to d_model
        → TransformerEncoder (causal mask)
        → Post-FC layers
        → 1 logit per position (binary: has wave V?)
    """
    max_seq_len = 38
    META_FEATURES = 2   # freq + dB
    WAVEFORM_LEN = 256  # raw waveform points

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        cnn_output_dim = config.get("cnn_output_dim", 32)
        pos_embedding_dim = config.get("pos_embedding_dim", 8)
        dropout = config.get("dropout", 0.1)
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # ── ABRWaveformCNNEncoder ──
        # GroupNorm (not BatchNorm) so small sessions (~7 waveforms) don't
        # corrupt the normalization statistics.
        # Dilated convolutions capture multi-scale Wave V morphology.
        # LayerNorm on the output embedding adds stability.
        self.cnn_encoder = ABRWaveformCNNEncoder(
            waveform_length=self.WAVEFORM_LEN,
            embed_dim=cnn_output_dim,
            dropout=dropout,
        )

        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)

        # Combined input: CNN embedding + metadata (freq, dB) + positional
        combined_dim = cnn_output_dim + self.META_FEATURES + pos_embedding_dim

        pre_layers = []
        prev_dim = combined_dim
        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        self.projection = nn.Linear(prev_dim, config["transformer_dim"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["transformer_dim"],
            nhead=config["transformer_heads"],
            dim_feedforward=config["transformer_ff"],
            dropout=dropout,
            activation=activation,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["transformer_layers"]
        )

        post_layers = []
        prev_dim = config["transformer_dim"]
        for h in config.get("post_fc_layers", []):
            post_layers.append(nn.Linear(prev_dim, h))
            post_layers.append(act_fn())
            post_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.post_fc = nn.Sequential(*post_layers)

        self.output = nn.Linear(prev_dim, 1)  # binary output

    def _encode_sequence(self, seq):
        """
        Encode a full session through CNN + causal Transformer.

        Args:
            seq: (seq_len, 258) — [freq, dB, value_0..value_255]

        Returns:
            logits: (seq_len,) — one logit per position
        """
        device = seq.device
        seq_len = seq.size(0)

        meta = seq[:, :self.META_FEATURES]          # (seq_len, 2) — freq + dB
        waveform = seq[:, self.META_FEATURES:]       # (seq_len, 256)

        # CNN encode — mean-center each waveform first (preserves amplitude
        # scale but removes DC bias), then pass through ABRWaveformCNNEncoder
        waveform = waveform.unsqueeze(1)                              # (seq_len, 1, 256)
        waveform = self.cnn_encoder.normalize_waveform(waveform)     # mean-center
        cnn_features = self.cnn_encoder(waveform)                    # (seq_len, cnn_output_dim)

        # Positional embeddings
        positions = torch.arange(seq_len, device=device)
        pos_embeds = self.pos_embedding(positions)   # (seq_len, pos_dim)

        # Combine all features
        x = torch.cat([cnn_features, meta, pos_embeds], dim=-1)
        x = self.pre_fc(x)
        x = self.projection(x)
        x = x.unsqueeze(1)  # (seq_len, batch=1, d_model)

        # Causal mask — position t can only attend to positions 0..t
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)

        x = self.post_fc(x)
        logits = self.output(x).squeeze(-1).squeeze(-1)  # (seq_len,)
        return logits

    def forward_train(self, seq, labels, sampling_prob=0.0):
        # sampling_prob ignored — non-autoregressive CNN-Transformer
        logits = self._encode_sequence(seq)
        return logits, labels

    def forward_test(self, seq):
        logits = self._encode_sequence(seq)
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()
        return predictions, logits
