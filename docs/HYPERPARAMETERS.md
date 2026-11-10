# Hyperparameters

Machine-readable configs live in `configs/`. Summary:

## Stimulus-change task (`configs/stimulus_change/`)
| Model | Key hyperparameters |
|---|---|
| DNN | pre-FC [16], hidden [64,64,32], post-FC [32], dropout 0.1, leaky ReLU, batch norm, lr 1e-4, 60 epochs |
| LSTM (CNN hybrid) | CNN embed 96, pool_k 8, LSTM hidden 128 × 4 layers, post-FC [128,64], dropout 0.4, label/pos embed 32, lr 5e-4, 60 epochs, linear sampling |
| LSTM (CNN, 23 features) | CNN embed 96, hidden 256 × 3 layers, post-FC [64], dropout 0.15, embeds 24, lr 5e-4, 60 epochs |
| Transformer | pre-FC [256,128,64], d_model 256, 8 heads, FF 256, 3 layers, post-FC [64], dropout 0.1, embeds 64, lr 1e-4, 60 epochs |
| Two-head CNN-Transformer | CNN embed 96, pool_k 8, d_model 256, 4 heads, FF 256, 6 layers, post-FC [128,64], dropout 0.15, 12 action classes (stop idx 13, repeat idx 7) |
| Random | seed 42 |

## Wave V task (`configs/wave_5/`)
| Model | Key hyperparameters |
|---|---|
| DNN | pre-FC [64], hidden [64,64,32], post-FC [64,32], dropout 0.1, lr 1e-3, 25 epochs |
| LSTM | pre-FC [64], hidden 64 × 2 layers, autoregressive, dropout 0.1 |

Grid-search spaces are defined in the `configs/` Python modules under `src/models/*/**/configs/`.
