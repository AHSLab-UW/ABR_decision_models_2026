import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import torch


class BaseResultsAggregator(ABC):
    def __init__(self, logger_class):
        self.logger_class = logger_class
        self.loggers = {}
        self.is_summarized = False
        self.history = {}  # {metric_name: [list_of_values_per_x_label]}
        self.x_labels = []

    @abstractmethod
    def get_key(self, **kwargs):
        pass

    @abstractmethod
    def format_label(self, key):
        pass

    def update(self, pred, label, logits, **kwargs):
        key = self.get_key(**kwargs)
        if key not in self.loggers:
            self.loggers[key] = self.logger_class()

        self.loggers[key].update_sequential(pred, label, logits, **kwargs)

    def summarize_results(self):
        sorted_keys = sorted(self.loggers.keys())
        self.x_labels = [self.format_label(k) for k in sorted_keys]
        self.history = {}

        for key in sorted_keys:
            logger = self.loggers[key]
            logger.summarize_results()  # Calculate internal stats

            # Pull metrics dynamically from the logger
            metrics = logger.get_plot_metrics()
            for name, value in metrics.items():
                if name not in self.history:
                    self.history[name] = []
                self.history[name].append(value)

        self.is_summarized = True

    def show_plot(self, title="Model Performance", save_path=None):
        if not self.is_summarized:
            self.summarize_results()

        plt.figure(figsize=(10, 6))
        # Plot every metric found in history with same style but different colors
        for label, values in self.history.items():
            plt.plot(self.x_labels, values, marker='o', linestyle='-', linewidth=2, label=label)

        plt.title(title)
        plt.xlabel("Progression / Group")
        plt.ylabel("Score (%)")
        plt.ylim(0, 105)  # Extra 5% for legend headroom
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()
        plt.close()

    def get_binned_results_dict(self):
        if not self.is_summarized:
            self.summarize_results()

        return {
            self.format_label(k): logger.get_results()
            for k, logger in self.loggers.items()
        }

    @abstractmethod
    def resolve_batch_keys(self, **kwargs):
        """Must return a tensor of keys corresponding to the batch size."""
        pass

    def update_batch(self, preds, labels, logits, **kwargs):
        batch_keys = self.resolve_batch_keys(**kwargs)

        unique_keys = torch.unique(batch_keys).tolist()

        for key in unique_keys:
            if key not in self.loggers:
                self.loggers[key] = self.logger_class()

            # 2. Mask the data for this group
            mask = (batch_keys == key)

            # 3. Dispatch to the sub-logger
            self.loggers[key].update_batch(
                preds[mask],
                labels[mask],
                logits[mask],
                **kwargs
            )