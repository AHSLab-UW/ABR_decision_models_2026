import torch
import torch.nn as nn
from src.common.model.api.base_model import ConfigurableModelBase


class ConfigurableDNNModel(ConfigurableModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        input_dim = config["input_size"]
        output_dim = config["output_size"]
        dropout = config.get("dropout", 0.1)

        activation_name = config.get("activation", "relu").lower()
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh
        }
        act_fn = activations.get(activation_name, nn.ReLU)

        self.use_bn = config.get("use_batch_norm", True)

        def build_block(in_f, out_f):
            block = [nn.Linear(in_f, out_f)]
            if self.use_bn:
                block.append(nn.BatchNorm1d(out_f))
            block.append(act_fn())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            return block

        pre_layers = []
        prev_dim = input_dim
        for h in config.get("pre_fc_layers", []):
            pre_layers.extend(build_block(prev_dim, h))
            prev_dim = h
        self.pre_fc = nn.Sequential(*pre_layers)

        dnn_layers = []
        for h in config.get("hidden_layers", [128, 128]):
            dnn_layers.extend(build_block(prev_dim, h))
            prev_dim = h
        self.main_dnn = nn.Sequential(*dnn_layers)

        post_layers = []
        for h in config.get("post_fc_layers", []):
            post_layers.extend(build_block(prev_dim, h))
            prev_dim = h
        self.post_fc = nn.Sequential(*post_layers)

        self.output = nn.Linear(prev_dim, output_dim)

    def forward_step(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.pre_fc(x)
        x = self.main_dnn(x)
        x = self.post_fc(x)
        return self.output(x)

    def forward_train(self, seq, labels, sampling_prob=0.0):
        logits = self.forward_step(seq)
        return logits, labels

    def forward_test(self, seq):
        logits = self.forward_step(seq)
        predictions = torch.argmax(logits, dim=-1)
        return predictions, logits