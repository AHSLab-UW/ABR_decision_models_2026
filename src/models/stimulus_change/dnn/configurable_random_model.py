import torch
import torch.nn as nn

from src.common.model.api.sequential_model import ConfigurableSequentialModel


class ConfigurableRandomModel(ConfigurableSequentialModel):
    """
    Dummy baseline model — ignores input entirely and returns
    uniformly random class predictions. Used to establish a floor
    for accuracy metrics (top-1, top-3, +-10dB, etc) that any real
    model should be expected to beat.

    Intended for testing_loop_sequential: forward_test returns the
    1-D logits of the last timestep. Only inputs.size(0) and
    inputs.device are read — never feature values.

    Randomness comes from a private generator seeded once at init,
    so every call draws fresh values (predictions don't depend on
    sequence position) and the global torch RNG is left untouched.
    Instantiate a fresh model per evaluated split for reproducibility.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_size = config["output_size"]

        # Dummy parameter so optimizer/param-count utilities
        # (e.g. print_model_architecture) don't break on an empty model.
        self._unused = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.seed = config['seed']
        self.gen = torch.Generator().manual_seed(self.seed)

    def _random_logits(self, seq):
        logits = torch.rand(seq.size(0), self.output_size, generator=self.gen)
        return logits.to(seq.device)

    def forward_train(self, seq, labels, sampling_prob=0.0):
        logits = self._random_logits(seq)
        return logits, labels.clone().detach().to(seq.device)

    def forward_test(self, seq):
        logits = self._random_logits(seq)
        # derive preds from the same logits so top-1 and top-k agree
        preds = torch.argmax(logits, dim=-1)
        return preds, logits[-1]
