import torch
import torch.nn as nn

from src.common.model.api.sequential_model import ConfigurableSequentialModel
from src.common.model.waveform_cnn_encoder import ABRWaveformCNNEncoder


class ConfigurableNoPrevPredsCNNLSTMModel(ConfigurableSequentialModel):
    max_seq_len = 64
    distinct_frequencies = 5

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. CNN Encoder Setup
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

        # 2. Meta Features & Layer Norm Setup
        # Explicitly save meta_dim so we can slice the seq tensor later
        self.meta_dim = config.get("input_size", config.get("meta_size", 12))
        combined_feature_dim = cnn_embed_dim + self.meta_dim

        self.input_layer_norm = nn.LayerNorm(combined_feature_dim)

        # Removed label_embedding_dim since we don't feed previous predictions
        pos_embedding_dim = config.get("pos_embedding_dim", 16)
        output_dim = config["output_size"]
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # 3. Pre-FC Layer
        pre_layers = []
        prev_dim = combined_feature_dim + pos_embedding_dim

        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        # 4. Positional Embedding & LSTM Setup
        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=prev_dim,
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_layers"],
            dropout=dropout if config["lstm_layers"] > 1 else 0.0,
        )

        # 5. Post-FC & Output Setup
        post_layers = []
        prev_dim = config["lstm_hidden_size"]
        for h in config.get("post_fc_layers", []):
            post_layers.append(nn.Linear(prev_dim, h))
            post_layers.append(act_fn())
            post_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.post_fc = nn.Sequential(*post_layers)

        self.output = nn.Linear(prev_dim, output_dim)

    def _encode_sequence(self, seq):
        """
        Vectorized processing for the entire combined sequence (Meta + Wave).
        """
        device = seq.device
        T = seq.size(0)

        # --- Extract meta and waveform ---
        meta = seq[:, :self.meta_dim]
        wave = seq[:, self.meta_dim:]

        # Unsqueeze channel dimension for CNN: [T, 256] -> [T, 1, 256]
        wave = wave.unsqueeze(1)

        # --- CNN Processing ---
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_emb = self.cnn_encoder(wave_norm)  # [T, cnn_embed_dim]

        # --- Combine, Normalize, and add Position ---
        x_combined = torch.cat([cnn_emb, meta], dim=-1)  # [T, combined_dim]
        x_combined = self.input_layer_norm(x_combined)

        pos_indices = torch.arange(T, device=device)
        pos_embed = self.pos_embedding(pos_indices)  # [T, pos_embedding_dim]

        # Final input to Pre-FC
        x = torch.cat([x_combined, pos_embed], dim=-1)

        x = self.pre_fc(x)  # [T, pre_fc_dim]

        # Add batch dimension for LSTM: [T, pre_fc_dim] -> [T, 1, pre_fc_dim]
        x = x.unsqueeze(1)

        # --- LSTM & Output ---
        lstm_out, (h, c) = self.lstm(x)  # [T, 1, lstm_hidden_size]

        hidden = self.post_fc(lstm_out)  # [T, 1, post_fc_dim]
        logits = self.output(hidden)  # [T, 1, output_dim]

        # Remove the dummy batch dimension
        return logits.squeeze(1)  # [T, output_dim]

    def forward_train(self, seq, labels, sampling_prob=0.0):
        # Process the entire sequence simultaneously
        logits = self._encode_sequence(seq)  # [T, output_dim]

        return logits, labels.clone().detach().to(seq.device)

    def forward_test(self, seq):
        # Inference is completely vectorized
        logits = self._encode_sequence(seq)  # [T, output_dim]

        predictions = torch.argmax(logits, dim=-1)  # [T]
        last_logits = logits[-1]  # [output_dim]

        return predictions, last_logits