from abc import ABC, abstractmethod

import pandas as pd


class BaseResultsLogger(ABC):
    @abstractmethod
    def update_sequential(self, pred, label, logits, **kwargs):
        pass

    @abstractmethod
    def update_batch(self, preds, labels, logits, **kwargs):
        pass

    @abstractmethod
    def summarize_results(self) -> float:
        pass

    @abstractmethod
    def get_cm(self):
        pass

    @abstractmethod
    def get_results(self) -> dict:
        pass

    @abstractmethod
    def get_empty_results_dict(self) -> dict:
        pass

    def get_true_labels_preds(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_accuracy(self) -> float:
        pass

    def is_overpredicting(self) -> bool:
        pass

    def get_plot_metrics(self) -> dict:
        return {"Accuracy": self.get_accuracy()}