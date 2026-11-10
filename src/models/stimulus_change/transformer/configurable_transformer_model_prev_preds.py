import torch
import torch.nn as nn

from src.common.model.api.sequential_model import ConfigurableSequentialModel


class ConfigurableTransformerPrevPredsModel(ConfigurableSequentialModel):
    max_seq_len = 64
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        input_dim = config["input_size"]
        output_dim = config["output_size"]
        embedding_dim = config.get("label_embedding_dim", 16)
        pos_embedding_dim = config.get("pos_embedding_dim", 16)
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
        transformer_input_dim = prev_dim

        self.class_embedding = nn.Embedding(output_dim + 1, embedding_dim)
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

    def forward_step(self, seq_so_far):
        x = self.pre_fc(seq_so_far)
        x = self.embedding(x).unsqueeze(1)  # (seq_len, batch=1, d_model)

        seq_len = x.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        x = self.transformer(x, mask=mask)

        x = self.post_fc(x)
        logits = self.output(x[-1, 0])
        return logits

    def forward_train(self, seq, labels, sampling_prob=0.0):
        device = seq.device
        num_classes = self.output.out_features

        outputs, labels_out = [], []
        prev_class = num_classes
        seq_so_far = []

        for t in range(len(seq)):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class))
            pos_embed = self.pos_embedding(torch.tensor(t))
            x_t = torch.cat([seq[t], prev_class_embed, pos_embed], dim=0)
            seq_so_far.append(x_t)
            seq_tensor = torch.stack(seq_so_far)
            logits = self.forward_step(seq_tensor)

            outputs.append(logits)
            labels_out.append(labels[t])

            if torch.rand(1).item() < sampling_prob:
                pred_class = torch.argmax(logits)
            else:
                pred_class = labels[t]

            prev_class = pred_class

        return torch.stack(outputs), torch.tensor(labels_out, device=device)

    def forward_test(self, seq):
        num_classes = self.output.out_features
        seq_len = seq.size(0)

        prev_class = num_classes
        seq_so_far = []
        predictions = []
        last_logits = None

        for t in range(seq_len):
            prev_class_embed = self.class_embedding(torch.tensor(prev_class))
            pos_embed = self.pos_embedding(torch.tensor(t))
            x_t = torch.cat([seq[t], prev_class_embed, pos_embed], dim=0)
            seq_so_far.append(x_t)
            seq_tensor = torch.stack(seq_so_far)
            logits = self.forward_step(seq_tensor)

            pred = torch.argmax(logits).item()
            predictions.append(pred)

            prev_class = pred
            last_logits = logits

        return torch.tensor(predictions, device=seq.device), last_logits
