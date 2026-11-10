import torch

from src.common.model.results.api.base_results_aggregator import BaseResultsAggregator


class FrequencyResults(BaseResultsAggregator):
    def resolve_batch_keys(self, freqs_log2, **kwargs):
        return torch.round(2 ** freqs_log2).int()

    def format_label(self, key):
        return f"{key} Hz"

    def get_key(self, freq_log2, **kwargs):  # For sequential mode
        return round(2 ** freq_log2)