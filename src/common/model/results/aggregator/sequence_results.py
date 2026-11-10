import math

from src.common.model.results.api.base_results_aggregator import BaseResultsAggregator


class SequenceResults(BaseResultsAggregator):
    def __init__(self, logger_class, num_bins=10):
        super().__init__(logger_class)
        self.num_bins = num_bins

    def get_key(self, current_length, total_length, **kwargs):
        norm_len = current_length / total_length
        idx = int(math.ceil(norm_len * self.num_bins)) - 1
        return max(0, min(idx, self.num_bins - 1))

    def format_label(self, key):
        upper_pct = int(((key + 1) / self.num_bins) * 100)
        return f"{upper_pct}%"

    def resolve_batch_keys(self, current_length, total_length, **kwargs):
        raise Exception("Sequence results should not be batched!")