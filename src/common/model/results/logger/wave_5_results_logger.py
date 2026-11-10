import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score

from src.common.model.results.api.base_results_logger import BaseResultsLogger


class Wave5ClassificationResultsLogger(BaseResultsLogger):
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.results = {}

    def update_sequential(self, pred, label, logits, **kwargs):
        p_val = pred[-1].item()
        l_val = label[-1].item()
        prob = torch.sigmoid(logits[-1]).item()

        self.correct += (1 if p_val == l_val else 0)
        self.total += 1

        self.predictions.append(p_val)
        self.true_labels.append(l_val)
        self.probabilities.append(prob)

    def update_batch(self, preds, labels, logits, **kwargs):
        labels = labels.float().view(-1)
        preds = preds.view(-1)
        probs = torch.sigmoid(logits).view(-1)

        self.total += labels.size(0)
        self.correct += (preds == labels).sum().item()

        self.predictions.extend(preds.cpu().tolist())
        self.true_labels.extend(labels.cpu().tolist())
        self.probabilities.extend(probs.cpu().tolist())

    def get_true_labels_preds(self) -> pd.DataFrame:
        return pd.DataFrame({'true': self.true_labels, 'predictions': self.predictions})

    def summarize_results(self):
        if self.total == 0:
            return 0.0

        accuracy = (self.correct / self.total * 100)

        precision = precision_score(self.true_labels, self.predictions, zero_division=0)
        recall = recall_score(self.true_labels, self.predictions, zero_division=0)
        f1 = f1_score(self.true_labels, self.predictions, zero_division=0)

        try:
            auc_roc = roc_auc_score(self.true_labels, self.probabilities)
        except ValueError:
            auc_roc = 0.5

        pred_1_ratio = sum(self.predictions) / len(self.predictions) if len(self.predictions) > 0 else 0

        print(f"Total Samples: {self.total}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f} (Accuracy of positive predictions)")
        print(f"Recall: {recall:.4f} (Rate of finding actual positives)")
        print(f"F1 Score: {f1:.4f} (Harmonic mean of Precision & Recall)")
        print(f"AUC-ROC: {auc_roc:.4f}\n")

        if pred_1_ratio > 0.95 or pred_1_ratio < 0.05:
            print(f"Model predicted Class 1 (Wave 5) {pred_1_ratio * 100:.2f}%")
            print("The model is stuck predicting a single class")
            accuracy = 0.0

        self.results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "pred_1_ratio": pred_1_ratio
        }

        return accuracy

    def get_cm(self):
        return confusion_matrix(y_true=self.true_labels, y_pred=self.predictions)

    def get_results(self):
        return self.results

    def is_overpredicting(self) -> bool:
        pred_1_ratio = sum(self.predictions) / len(self.predictions) if len(self.predictions) > 0 else 0
        return pred_1_ratio > 0.95 or pred_1_ratio < 0.05

    def get_empty_results_dict(self):
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.5,
            "pred_1_ratio": 0.0
        }

    def get_accuracy(self) -> float:
        return (self.correct / self.total * 100)
