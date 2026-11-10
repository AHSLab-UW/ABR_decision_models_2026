import torch
import torch.nn as nn

from src.common.model.api.sequential_model import ConfigurableSequentialModel


class ConfigurablePrevPredsLSTMModel(ConfigurableSequentialModel):
    max_seq_len = 64
    distinct_frequencies = 5
    def __init__(self, config):
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
        prev_dim = input_dim + embedding_dim + pos_embedding_dim #position parameter
        for h in config["pre_fc_layers"]:
            pre_layers.append(nn.Linear(prev_dim, h))
            pre_layers.append(act_fn())
            pre_layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        lstm_input_dim = prev_dim

        self.class_embedding = nn.Embedding(output_dim + 1, embedding_dim)
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
        for h in config["post_fc_layers"]:
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
        num_classes = self.output.out_features

        outputs, labels_out = [], []

        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros_like(h)
        prev_class = num_classes

        for t in range(len(seq)):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))
            x_t = torch.cat([seq[t], prev_class_embed, pos_embed], dim=0)
            logits, h, c = self.forward_step(x_t, h, c)

            outputs.append(logits[0, 0])
            labels_out.append(labels[t])

            if torch.rand(1).item() < sampling_prob:
                pred_class = torch.argmax(logits[0, 0]).item()
            else:
                pred_class = labels[t].item()

            prev_class = pred_class

        return torch.stack(outputs), torch.tensor(labels_out, device=device)

    def forward_test(self, x):
        num_classes = self.output.out_features
        seq_len = x.size(0)

        prev_class = num_classes
        h = torch.zeros(self.lstm.num_layers,
                        1, self.lstm.hidden_size)
        c = torch.zeros_like(h)

        predictions = []
        last_logits = None

        for t in range(seq_len):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class))
            pos_embed = self.pos_embedding(torch.tensor(t))
            x_t = torch.cat([x[t], prev_class_embed, pos_embed], dim=0)

            logits, h, c = self.forward_step(x_t, h, c)

            pred = torch.argmax(logits, dim=2)[0, 0].item()
            predictions.append(pred)

            prev_class = pred
            last_logits = logits[0, 0]

        return torch.tensor(predictions), last_logits