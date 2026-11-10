from abc import ABC, abstractmethod
import torch.nn as nn


class ConfigurableModelBase(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward_train(self, seq, labels, sampling_prob=0.0):
        pass

    @abstractmethod
    def forward_test(self, seq):
        pass