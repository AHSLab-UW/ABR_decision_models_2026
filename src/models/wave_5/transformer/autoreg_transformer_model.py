import torch
import torch.nn as nn

from src.common.model.api.base_model import ConfigurableModelBase


class Wave5AutoregTransformerModel(ConfigurableModelBase):
    """
    Autoregressive causal Transformer for binary Wave V classification.

    Unlike the non-autoregressive Wave5TransformerModel, this model feeds
    the PREVIOUS prediction back as a class embedding at each timestep —
    mirroring how a clinician's interpretation of earlier waveforms informs
    later decisions.

    At each timestep t, the model sees:
        - Input features (12 PCA+freq+dB) for positions 0..t
        - Class embedding of the previous prediction (or start token at t=0)
        - Positional embedding
    Then runs the full causal transformer on [0..t] and predicts from position t.

    Training uses scheduled sampling: with probability `sampling_prob`, the model
    uses its own prediction as prev_class; otherwise it uses the ground-truth label.

    Architecture:
        [Input features + class embedding + positional embedding]
        → Pre-FC layers
        → Linear projection to d_model
        → TransformerEncoder (causal mask — past only)
        → Post-FC layers
        → 1 logit at last position (binary: has wave V?)
    """
    max_seq_len = 38

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        input_dim = config["input_size"]               # 12
        embedding_dim = config.get("label_embedding_dim", 8)
        pos_embedding_dim = config.get("pos_embedding_dim", 8)
        dropout = config.get("dropout", 0.1)
        activation = config.get("activation", "relu")
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh

        # Class embedding: 0 = no wave V, 1 = wave V, 2 = start token
        self.class_embedding = nn.Embedding(3, embedding_dim)
        self.pos_embedding = nn.Embedding(self.max_seq_len, pos_embedding_dim)

        # Pre-FC: input_features + class_embed + pos_embed
        pre_layers = []
        prev_dim = input_dim + embedding_dim + pos_embedding_dim
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

    def _forward_step(self, seq_so_far):
        """Run causal transformer on accumulated sequence, predict from last position."""
        x = self.pre_fc(seq_so_far)
        x = self.projection(x)
        x = x.unsqueeze(1)  # (seq_len, batch=1, d_model)

        seq_len = x.size(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)

        x = self.post_fc(x)
        logits = self.output(x[-1, 0])  # predict from last position only
        return logits.squeeze(-1)       # scalar

    def forward_train(self, seq, labels, sampling_prob=0.0):
        """
        Autoregressive training with scheduled sampling.
        
        Args:
            seq:           (seq_len, 12) input features
            labels:        (seq_len,) integer labels (0 or 1)
            sampling_prob: probability of using own prediction vs ground truth
        
        Returns:
            (logits_tensor, labels_tensor) both shape (seq_len,)
        """
        device = seq.device
        outputs, labels_out = [], []
        prev_class = 2  # start token
        seq_so_far = []

        for t in range(len(seq)):
            prev_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))
            x_t = torch.cat([seq[t], prev_embed, pos_embed], dim=0)
            seq_so_far.append(x_t)

            seq_tensor = torch.stack(seq_so_far)
            logits = self._forward_step(seq_tensor)

            outputs.append(logits)
            labels_out.append(labels[t])

            # Scheduled sampling: use own prediction or ground truth
            if torch.rand(1).item() < sampling_prob:
                pred_class = int((torch.sigmoid(logits) >= 0.5).float().item())
            else:
                pred_class = labels[t].item()

            prev_class = pred_class

        return torch.stack(outputs), torch.tensor(labels_out, device=device)

    def forward_test(self, seq):
        """
        Autoregressive inference — step by step, feeding own predictions back.
        
        Returns:
            (predictions_tensor, logits_tensor) both shape (seq_len,)
        """
        device = seq.device
        prev_class = 2  # start token
        seq_so_far = []
        predictions = []
        all_logits = []

        for t in range(seq.size(0)):
            prev_embed = self.class_embedding(torch.tensor(prev_class, device=device))
            pos_embed = self.pos_embedding(torch.tensor(t, device=device))
            x_t = torch.cat([seq[t], prev_embed, pos_embed], dim=0)
            seq_so_far.append(x_t)

            seq_tensor = torch.stack(seq_so_far)
            logits = self._forward_step(seq_tensor)

            pred = int((torch.sigmoid(logits) >= 0.5).float().item())
            predictions.append(float(pred))
            all_logits.append(logits)

            prev_class = pred

        return torch.tensor(predictions, device=device), torch.stack(all_logits)

