import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.common.model.model_util_common import flatten_labels
from src.common.model.results.api.base_results_logger import BaseResultsLogger


class StimulusChangeResults(BaseResultsLogger):
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.pm10db_correct, self.pm5db_correct, self.pm15db_correct = 0, 0, 0
        self.quasi_total = 0
        self.top3_correct, self.top5_correct = 0, 0
        self.top3_repeats, self.top5_repeats = 0, 0
        self.predictions, self.true_labels = [], []
        self.results = {}
        self.predicted_frequency = {}

    def update_sequential(self, pred, label, logits, **kwargs):
        reverse_map = kwargs.get('reverse_map')
        if reverse_map is None:
            raise ValueError("Need reverse_map")

        p_val = pred[-1].item()
        l_val = label[-1].item()

        top3 = torch.topk(logits, k=3).indices.tolist()
        top5 = torch.topk(logits, k=5).indices.tolist()

        self.top3_correct += int(l_val in top3)
        self.top5_correct += int(l_val in top5)
        self.top3_repeats += int(7 in top3)
        self.top5_repeats += int(7 in top5)

        self.correct += (1 if p_val == l_val else 0)
        self.total += 1

        correct_mapped = reverse_map[l_val]
        pred_mapped = reverse_map[p_val]

        if pred_mapped not in self.predicted_frequency:
            self.predicted_frequency[pred_mapped] = 0
        self.predicted_frequency[pred_mapped] += 1

        exclude_strs = ['Stop', '>', '<']
        if not any(ex in correct_mapped or ex in pred_mapped for ex in exclude_strs):
            self.quasi_total += 1

            value = correct_mapped.split(" ")[0]
            value_pred = pred_mapped.split(" ")[0]

            if "Repeat" in value: value = "0"
            if "Repeat" in value_pred: value_pred = "0"

            diff = abs(int(value) - int(value_pred))

            if diff <= 5: self.pm5db_correct += 1
            if diff <= 10: self.pm10db_correct += 1
            if diff <= 15: self.pm15db_correct += 1

        self.predictions.append(p_val)
        self.true_labels.append(l_val)

    def update_batch(self, preds, labels, logits, **kwargs):
        reverse_map = kwargs.get('reverse_map')
        if reverse_map is None:
            raise ValueError("Need reverse_map")

        top3_inds = torch.topk(logits, k=3, dim=1).indices
        top5_inds = torch.topk(logits, k=5, dim=1).indices

        self.top3_correct += (top3_inds == labels.unsqueeze(1)).any(dim=1).sum().item()
        self.top5_correct += (top5_inds == labels.unsqueeze(1)).any(dim=1).sum().item()
        self.top3_repeats += (top3_inds == 7).any(dim=1).sum().item()
        self.top5_repeats += (top5_inds == 7).any(dim=1).sum().item()

        self.total += labels.size(0)
        self.correct += (preds == labels).sum().item()

        preds_cpu = preds.tolist()
        labels_cpu = labels.tolist()

        for p_val, l_val in zip(preds_cpu, labels_cpu):
            p_str = reverse_map[p_val]
            t_str = reverse_map[l_val]
            if p_str not in self.predicted_frequency:
                self.predicted_frequency[p_str] = 0
            self.predicted_frequency[p_str] += 1

            if not any(ex in p_str or ex in t_str for ex in ['Stop', '>', '<']):
                self.quasi_total += 1

                val_p = "0" if "Repeat" in p_str else p_str.split(" ")[0]
                val_t = "0" if "Repeat" in t_str else t_str.split(" ")[0]

                try:
                    diff = abs(int(val_t) - int(val_p))
                    if diff <= 5: self.pm5db_correct += 1
                    if diff <= 10: self.pm10db_correct += 1
                    if diff <= 15: self.pm15db_correct += 1
                except ValueError:
                    pass

        self.predictions.extend(preds_cpu)
        self.true_labels.extend(labels_cpu)

    def summarize_results(self):
        if self.total == 0:
            return 0.0

        accuracy = self.correct / self.total * 100
        top3_acc = self.top3_correct / self.total * 100
        top5_acc = self.top5_correct / self.total * 100
        top3_rep_acc = self.top3_repeats / self.total * 100
        top5_rep_acc = self.top5_repeats / self.total * 100

        qt = self.quasi_total if self.quasi_total > 0 else 1
        pm5_acc = 100 * (self.pm5db_correct / qt)
        pm10_acc = 100 * (self.pm10db_correct / qt)
        pm15_acc = 100 * (self.pm15db_correct / qt)

        self.predicted_frequency = {k: (v/self.total) for k,v in self.predicted_frequency.items()}

        print(f"Fraction: {self.correct}/{self.total}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Top-3 Accuracy: {top3_acc:.2f}%")
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
        print(f"Top-3 Repeats: {top3_rep_acc:.2f}%")
        print(f"Top-5 Repeats: {top5_rep_acc:.2f}%")
        print(f"Within 5 dB Accuracy: {self.pm5db_correct}/{self.quasi_total} = {pm5_acc:.2f}%")
        print(f"Within 10 dB Accuracy: {self.pm10db_correct}/{self.quasi_total} = {pm10_acc:.2f}%")
        print(f"Within 15 dB Accuracy: {self.pm15db_correct}/{self.quasi_total} = {pm15_acc:.2f}%")
        print(f"Predicted Class Distribution: {self.predicted_frequency}")

        self.results = {
            "accuracy": accuracy,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
            "top3_rep_acc": top3_rep_acc,
            "top5_rep_acc": top5_rep_acc,
            "pm5db_correct": pm5_acc,
            "pm10db_correct": pm10_acc,
            "pm15db_correct": pm15_acc,
            "predicted_class_distribution": self.predicted_frequency
        }
        print("Length", len(self.predictions))

        return accuracy

    def get_cm(self):
        return confusion_matrix(y_true=flatten_labels(self.true_labels), y_pred=flatten_labels(self.predictions))

    def get_results(self):
        return self.results

    def get_empty_results_dict(self):
        return {
            "accuracy": 0.0,
            "top3_acc": 0.0,
            "top5_acc": 0.0,
            "top3_rep_acc": 0.0,
            "top5_rep_acc": 0.0,
            "pm5db_correct": 0.0,
            "pm10db_correct": 0.0,
            "pm15db_correct": 0.0,
            "predicted_class_distribution": {}
        }

    def get_accuracy(self) -> float:
        return self.correct / self.total * 100

    def is_overpredicting(self):
        # extra guard so we don't accidently store only repeats (kinda iffy on selection)
        return any(val > 0.90 for val in self.predicted_frequency.values())

    def get_plot_metrics(self) -> dict:
        return {
            "Accuracy": self.get_accuracy(),
            "Top-3 Accuracy": self.top3_correct / self.total * 100,
            "±10dB Accuracy": 100 * (self.pm10db_correct / self.quasi_total)
        }

    def get_true_labels_preds(self) -> pd.DataFrame:
        return pd.DataFrame({'true': self.true_labels, 'predictions': self.predictions})
