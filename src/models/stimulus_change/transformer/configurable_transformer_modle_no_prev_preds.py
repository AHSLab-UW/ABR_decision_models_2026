import torch
import torch.nn as nn

from src.common.model.api.sequential_model import ConfigurableSequentialModel


class ConfigurableNoPrevPredsTransformerModel(ConfigurableSequentialModel):
    max_seq_len = 38

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        input_dim = config["input_size"]
        output_dim = config["output_size"]

        pos_embedding_dim = config.get("pos_embedding_dim", 16)
        dropout = config.get("dropout", 0.1)
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        pre_layers = []
        prev_dim = input_dim + pos_embedding_dim
        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        transformer_input_dim = prev_dim

        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)

        self.embedding = nn.Linear(transformer_input_dim, config["transformer_dim"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["transformer_dim"],
            nhead=config["transformer_heads"],
            dim_feedforward=config["transformer_ff"],
            dropout=dropout,
            activation=activation
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

        self.output = nn.Linear(prev_dim, output_dim)

    def _encode_sequence(self, seq):
        device = seq.device
        T = seq.size(0)

        pos_indices = torch.arange(T, device=device)
        pos_embed = self.pos_embedding(pos_indices)  # [T, pos_embedding_dim]

        x = torch.cat([seq, pos_embed], dim=-1)  # [T, input_dim + pos_embedding_dim]

        x = self.pre_fc(x)
        x = self.embedding(x)

        x = x.unsqueeze(1)

        causal_mask = torch.triu(torch.ones(T, T, device=device) * float('-inf'), diagonal=1)

        x = self.transformer(x, mask=causal_mask)

        x = self.post_fc(x)
        logits = self.output(x)

        return logits.squeeze(1)  # [T, output_dim]

    def forward_train(self, seq, labels, sampling_prob=0.0):
        logits = self._encode_sequence(seq)

        return logits, labels.clone().detach().to(seq.device)

    def forward_test(self, seq):
        logits = self._encode_sequence(seq)

        predictions = torch.argmax(logits, dim=-1)  # [T]

        last_logits = logits[-1]

        return predictions, last_logits