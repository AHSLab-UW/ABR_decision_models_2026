import torch
import torch.nn as nn

from src.common.model.api.sequential_model import ConfigurableSequentialModel


class ConfigurableNoPrevPredsLSTMModel(ConfigurableSequentialModel):
    max_seq_len = 64
    distinct_frequencies = 5

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        pos_embedding_dim = config.get("pos_embedding_dim", 16)
        input_dim = config["input_size"]
        output_dim = config["output_size"]
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

        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=prev_dim,
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_layers"],
            dropout=dropout if config["lstm_layers"] > 1 else 0.0,
        )

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
        device = seq.device
        T = seq.size(0)

        pos_indices = torch.arange(T, device=device)
        pos_embed = self.pos_embedding(pos_indices)

        x = torch.cat([seq, pos_embed], dim=-1)
        #x = torch.cat([seq], dim=-1)

        x = self.pre_fc(x)

        x = x.unsqueeze(1)

        lstm_out, (h, c) = self.lstm(x)

        hidden = self.post_fc(lstm_out)
        logits = self.output(hidden)

        return logits.squeeze(1)

    def forward_train(self, seq, labels, sampling_prob=0.0):
        logits = self._encode_sequence(seq)

        return logits, labels.clone().detach().to(seq.device)

    def forward_test(self, seq):
        logits = self._encode_sequence(seq)

        predictions = torch.argmax(logits, dim=-1)

        last_logits = logits[-1]

        return predictions, last_logits