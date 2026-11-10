# Architecture

All models predict the next stimulus-level action in a sequential ABR threshold search (`output_size = 14` classes: dB adjustments, a repeat class and a stop class).

## Inputs per step
- Raw averaged ABR waveform, 256 samples (resampled to a common rate) → `[T, 1, 256]`
- Meta features (12): stimulus frequency (log2), last level, and 10 PCA components of waveform-derived features (PCA fit on the training split only)

## Components (`src/`)
| Component | Location | Description |
|---|---|---|
| Waveform CNN encoder | `src/common/model/waveform_cnn_encoder.py` | Dilated 1D conv blocks (Conv → GroupNorm → SiLU), adaptive average pooling to `pool_k` bins, linear projection to `cnn_embed_dim` |
| DNN baseline | `src/models/stimulus_change/dnn/` | Feed-forward on the 12 meta features, optional batch norm |
| LSTM | `src/models/stimulus_change/lstm/` | Stacked LSTM over the step sequence; optional CNN waveform branch; optional previous-prediction (autoregressive) embedding |
| Transformer | `src/models/stimulus_change/transformer/` | Pre-FC → projection + positional embedding → causal Transformer encoder → post-FC; non-autoregressive variants use causal masking only |
| Two-head transformer | `.../cnn_transformer/non_auto_transformer_model.py` | First head: STOP / REPEAT / ADJUST (3-class); action head: dB adjustment classes |
| Random baseline | `src/models/stimulus_change/dnn/configurable_random_model.py` | Seeded random predictor |

## Training / evaluation
- Sequence-level 80/10/10 train/test/holdout split; PCA and scalers fit on train only
- Cross-entropy with optional sampling strategy (`linear`, `none`) and ordinal loss for the action head
- Metrics: accuracy, per-frequency accuracy, per-sequence accuracy, confusion matrices (`src/common/model/results/`)
