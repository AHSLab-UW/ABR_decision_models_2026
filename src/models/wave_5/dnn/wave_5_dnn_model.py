import torch
import torch.nn as nn

from src.models.stimulus_change.dnn.configurable_dnn_model import ConfigurableDNNModel


class Wave5DNNModel(ConfigurableDNNModel):
    def __init__(self, config):
        config['output_size'] = 1
        super().__init__(config)

    def forward_test(self, seq):
        logits = self.forward_step(seq)

        probabilities = torch.sigmoid(logits)

        predictions = (probabilities >= 0.5).float().squeeze(-1)

        return predictions, logits