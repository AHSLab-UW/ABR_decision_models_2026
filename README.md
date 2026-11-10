# ABR_decision_models_2026

Code supporting the development and evaluation of DNN, LSTM, and Transformer models for auditory brainstem response (ABR) waveform quality classification and stimulus-decision prediction. This repository accompanies publication of simulation results for machine-learning-assisted ABR threshold search.

This repository exists for **reproducibility**: it contains architecture descriptions, hyperparameters, training/evaluation/preprocessing/figure-generation code, and pseudocode.

**Not included:** trained weights, proprietary feature-generation pipelines, deployment code, and — critically — any raw ABR waveforms or clinical data. See [docs/DATA_POLICY.md](docs/DATA_POLICY.md). Users must supply their own data under appropriate approvals.

## Funding and acknowledgments

This work was conducted under NIH STTR award **R41DC022808** between **Intelligent Hearing Systems** and the **University of Washington**. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

## Code organization

```
configs/            Hyperparameter JSON files (stimulus_change/, wave_5/)
docs/               Architecture, hyperparameters, pseudocode, data policy
src/
  common/           Data utilities, CNN encoder, model base classes, results loggers, plotting
  parse/            Dataset parsing, visualizations, architecture diagrams
  preprocess/       Action tables, feature grouping, PCA, preprocessing pipeline
  models/
    stimulus_change/  DNN / LSTM / Transformer models, configs, training entry points (impl/)
    wave_5/           Wave V detection models and training entry points
  utils/            Waveform sequence dataset helpers
```

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Hyperparameters: [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) and `configs/`
- Pseudocode: [docs/PSEUDOCODE.md](docs/PSEUDOCODE.md)

## Usage

```
pip install -r requirements.txt
# Place locally prepared tables under data/ (git-ignored), then, from the repo root:
python -m src.preprocess.preprocess_pipeline
python -m src.models.stimulus_change.impl.transformer.transformer_impl --epochs 60
```

Scripts write outputs (results, fitted PCA/scalers) under git-ignored `exported_features/`. Several scripts use paths relative to their own directory; run them from their folder if a module invocation fails.

## Contact

shenyi@uw.edu

## License

BSD 3-Clause. See [LICENSE](LICENSE).
