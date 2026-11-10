# Data and artifact policy

This repository contains code and configuration only. It must **never** contain:

- raw ABR waveforms (AEPASCII `.Txt`, XML exports, waveform tables) or any clinical / patient data
- derived per-record tables (action, feature, session, record, waveform tables), per-sample prediction dumps (`pred_spread.csv`)
- trained weights (`.pth`, `.pt`, `.onnx`) or fitted preprocessing artifacts (PCA / scaler `.joblib`, PCA component CSVs), which encode properties of the clinical data
- proprietary feature-generation pipelines
- deployment / inference-serving code

`.gitignore` blocks these file types and the `data/`, `exported_features/`, and logs directories. Reviewers should check every pull request against this list.

To reproduce results you must obtain data under your own approvals and place derived tables under `data/` locally (see README).
