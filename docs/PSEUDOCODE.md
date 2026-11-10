# Pseudocode

## Preprocessing
```
for each dataset in {site A, site B}:
    parse exported records -> session / record / waveform tables
    build action table: for consecutive records in a session, label the next-step
        action (dB change, repeat, stop)
    for each record at target frequencies {500,1000,2000,4000,8000} Hz:
        resample waveform to 4 kHz, crop/pad to 256 samples, drop NaN records
        compute waveform-derived features
    group records into per-session sequences
```

## Training
```
split sequences 80/10/10 (train/test/holdout) at the sequence level
fit PCA(10) and scaler on TRAIN sequences only; apply to all splits
for config in grid(config space):
    model <- build(config)
    for epoch in 1..E:
        for sequence in train:
            logits <- model(waveforms, meta)        # causal over time
            loss   <- CE(logits, actions) [+ ordinal loss on action head]
            update(model, loss)
    evaluate on test; keep best config by accuracy
evaluate best config on holdout
```

## Inference over a sequence (non-autoregressive, causal)
```
h_t <- CNN(waveform_t) ++ meta_t
z   <- TransformerEncoder(pos_emb(proj(h_1..T)), causal_mask)
action_t <- argmax head(z_t)    # STOP / REPEAT / ADJUST(+dB class)
```
