import torch
import torch.nn as nn

from src.common.model.api.base_model import ConfigurableModelBase


class Wave5TransformerModel(ConfigurableModelBase):
    """
    Causal Transformer encoder for binary Wave V classification.

    Processes a full clinical ABR session in one forward pass, but each position
    can only attend to itself and PAST positions (causal masking). This reflects
    real-time clinical use: when the audiologist records waveform t, waveforms
    t+1, t+2, ... haven't been recorded yet.

    NOT autoregressive — no previous-prediction embeddings, no scheduled sampling,
    no error propagation. Just causal attention over raw features.

    Architecture:
        Input features (12) + positional embedding
        → Pre-FC layers
        → Linear projection to d_model
        → TransformerEncoder (causal mask — past only)
        → Post-FC layers
        → 1 logit per position (binary: has wave V?)
    """
    max_seq_len = 38

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        input_dim = config["input_size"]
        pos_embedding_dim = config.get("pos_embedding_dim", 8)
        dropout = config.get("dropout", 0.1)
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)

        pre_layers = []
        prev_dim = input_dim + pos_embedding_dim
        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)
        transformer_input_dim = prev_dim

        self.projection = nn.Linear(transformer_input_dim, config["transformer_dim"])

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
        transformer_output_dim = config["transformer_dim"]

        post_layers = []
        prev_dim = transformer_output_dim
        for h in config.get("post_fc_layers", []):
            post_layers.append(nn.Linear(prev_dim, h))
            post_layers.append(act_fn())
            post_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.post_fc = nn.Sequential(*post_layers)

        self.output = nn.Linear(prev_dim, 1)  # binary output

    def _encode_sequence(self, seq):
        """Encode full sequence through transformer. Returns logits at every position."""
        device = seq.device
        seq_len = seq.size(0)

        positions = torch.arange(seq_len, device=device)
        pos_embeds = self.pos_embedding(positions)  # (seq_len, pos_dim)

        x = torch.cat([seq, pos_embeds], dim=-1)  # (seq_len, input_dim + pos_dim)
        x = self.pre_fc(x)
        x = self.projection(x)
        x = x.unsqueeze(1)  # (seq_len, batch=1, d_model)

        # causal mask — position t can only attend to positions 0..t
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=causal_mask)

        x = self.post_fc(x)
        logits = self.output(x).squeeze(-1).squeeze(-1)  # (seq_len,)
        return logits

    def forward_train(self, seq, labels, sampling_prob=0.0):
        # sampling_prob is ignored — non-autoregressive, no scheduled sampling
        logits = self._encode_sequence(seq)
        return logits, labels

    def forward_test(self, seq):
        logits = self._encode_sequence(seq)
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()
        return predictions, logits
