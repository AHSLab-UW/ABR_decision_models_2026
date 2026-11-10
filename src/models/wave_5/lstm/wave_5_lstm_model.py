import torch
import torch.nn as nn
from src.common.model.api.sequential_model import ConfigurableSequentialModel


class Wave5LSTMModel(ConfigurableSequentialModel):
    max_seq_len = 38

    def __init__(self, config):
        config['output_size'] = 1
        super().__init__(config)
        self.config = config

        embedding_dim = config.get("label_embedding_dim", 16)
        pos_embedding_dim = config.get("pos_embedding_dim", 16)
        input_dim = config["input_size"]
        output_dim = config["output_size"]
        dropout = config.get("dropout", 0.1)
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        pre_layers = []
        prev_dim = input_dim + embedding_dim + pos_embedding_dim
        for h in config.get("pre_fc_layers", []):
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        lstm_input_dim = prev_dim

        self.class_embedding = nn.Embedding(3, embedding_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_layers"],
            dropout=dropout if config["lstm_layers"] > 1 else 0.0,
        )

        lstm_output_dim = config["lstm_hidden_size"]
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

    def forward_train(self, seq, labels, sampling_prob=0.0):
        device = seq.device
        outputs, labels_out = [], []

        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)

        prev_class = 2

        for t in range(len(seq)):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))
            x_t = torch.cat([seq[t], prev_class_embed, pos_embed], dim=0)

            logits, h, c = self.forward_step(x_t, h, c)

            outputs.append(logits[0, 0, 0])
            labels_out.append(labels[t])

            if torch.rand(1).item() < sampling_prob:
                pred_class = 1 if torch.sigmoid(logits[0, 0, 0]).item() >= 0.5 else 0
            else:
                pred_class = int(labels[t].item())

            prev_class = pred_class

        return torch.stack(outputs), torch.tensor(labels_out, device=device, dtype=torch.float32)

    def forward_test(self, x):
        seq_len = x.size(0)
        device = x.device

        prev_class = 2  # Start Token
        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)

        predictions = []
        logits_list = []

        for t in range(seq_len):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))
            x_t = torch.cat([x[t], prev_class_embed, pos_embed], dim=0)

            logits, h, c = self.forward_step(x_t, h, c)
            logit_val = logits[0, 0, 0]

            prob = torch.sigmoid(logit_val).item()
            pred = 1.0 if prob >= 0.5 else 0.0

            predictions.append(pred)
            logits_list.append(logit_val)

            prev_class = int(pred)

        return torch.tensor(predictions, device=device, dtype=torch.float32), torch.stack(logits_list)