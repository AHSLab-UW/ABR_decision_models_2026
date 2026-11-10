import torch
import torch.nn as nn

from src.common.model.api.sequential_model import ConfigurableSequentialModel
from src.common.model.waveform_cnn_encoder import ABRWaveformCNNEncoder


class ConfigurablePrevPredsCNNLSTMModel(ConfigurableSequentialModel):
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
        # Explicitly save meta_dim so we can use it to slice the seq tensor later
        self.meta_dim = config.get("input_size", config.get("meta_size", 12))
        combined_feature_dim = cnn_embed_dim + self.meta_dim

        # LayerNorm is crucial here to balance the massive CNN outputs with the tiny scaled metadata
        self.input_layer_norm = nn.LayerNorm(combined_feature_dim)

        embedding_dim = config.get("label_embedding_dim", 16)
        pos_embedding_dim = config.get("pos_embedding_dim", 16)
        output_dim = config["output_size"]
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # 3. Pre-FC Layer (After CNN/Meta, Before LSTM)
        pre_layers = []
        prev_dim = combined_feature_dim + embedding_dim + pos_embedding_dim

        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        # 4. LSTM & Embeddings Setup
        self.class_embedding = nn.Embedding(output_dim + 1, embedding_dim)
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

    def forward_step(self, x_t, h, c):
        # x_t is a 1D tensor [features]. We need to unsqueeze it for the LSTM [1, 1, features]
        x_t = self.pre_fc(x_t)
        x_t = x_t.unsqueeze(0).unsqueeze(0)

        lstm_out, (h, c) = self.lstm(x_t, (h, c))
        hidden = self.post_fc(lstm_out)
        logits = self.output(hidden)
        return logits, h, c

    def forward_train(self, seq, labels, sampling_prob=0.0):
        device = seq.device
        num_classes = self.output.out_features
        T = seq.size(0)

        # --- Extract meta and waveform from the combined seq tensor ---
        meta = seq[:, :self.meta_dim]
        wave = seq[:, self.meta_dim:]

        # The CNN expects [T, 1, L], so we unsqueeze the channel dimension
        wave = wave.unsqueeze(1)

        # --- OPTIMIZATION: Process entire sequence through CNN at once ---
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_emb = self.cnn_encoder(wave_norm)  # [T, cnn_embed_dim]

        # Combine CNN and Meta, then LayerNorm
        x_combined = torch.cat([cnn_emb, meta], dim=-1)  # [T, combined_dim]
        x_combined = self.input_layer_norm(x_combined)

        outputs, labels_out = [], []

        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        prev_class = num_classes

        # --- Autoregressive LSTM Loop ---
        for t in range(T):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))

            # Combine current step features with previous prediction & position
            x_t = torch.cat([x_combined[t], prev_class_embed, pos_embed], dim=0)

            logits, h, c = self.forward_step(x_t, h, c)

            outputs.append(logits[0, 0])
            labels_out.append(labels[t])

            # Scheduled Sampling
            if torch.rand(1).item() < sampling_prob:
                pred_class = torch.argmax(logits[0, 0]).item()
            else:
                pred_class = labels[t].item()

            prev_class = pred_class

        return torch.stack(outputs), torch.tensor(labels_out, device=device)

    def forward_test(self, seq):
        device = seq.device
        num_classes = self.output.out_features
        T = seq.size(0)

        # --- Extract meta and waveform from the combined seq tensor ---
        meta = seq[:, :self.meta_dim]
        wave = seq[:, self.meta_dim:]

        # The CNN expects [T, 1, L], so we unsqueeze the channel dimension
        wave = wave.unsqueeze(1)

        # Process entire sequence through CNN
        wave_norm = self.cnn_encoder.normalize_waveform(wave)
        cnn_emb = self.cnn_encoder(wave_norm)

        x_combined = torch.cat([cnn_emb, meta], dim=-1)
        x_combined = self.input_layer_norm(x_combined)

        prev_class = num_classes
        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)

        predictions = []
        last_logits = None

        for t in range(T):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))

            x_t = torch.cat([x_combined[t], prev_class_embed, pos_embed], dim=0)

            logits, h, c = self.forward_step(x_t, h, c)

            # Logits shape from forward_step is [1, 1, classes]. Extract to [classes]
            logits_flat = logits[0, 0]
            pred = torch.argmax(logits_flat).item()
            predictions.append(pred)

            prev_class = pred
            last_logits = logits_flat

        return torch.tensor(predictions, device=device), last_logits