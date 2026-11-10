"""
Waveform Data Loader
====================
Utilities for loading raw waveforms and creating sequences with (wave, meta, y).
"""

import os
import numpy as np
import pandas as pd
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))


def normalize_meta(meta, include_pca=True):
    """Normalize metadata using only statistics available in the provided sequence."""
    meta = meta.copy()

    # Per-sequence z-score for last_level
    last_level = meta[:, 1:2]
    if last_level.std() > 0:
        meta[:, 1:2] = (last_level - last_level.mean()) / (last_level.std() + 1e-8)

    if include_pca and meta.shape[1] >= 12:
        # Normalize extracted/PCA features per feature dimension within the sequence.
        pca_feats = meta[:, 2:12]
        feat_std = pca_feats.std(axis=0)
        nonzero = feat_std > 0
        if nonzero.any():
            pca_feats[:, nonzero] = (
                pca_feats[:, nonzero] - pca_feats[:, nonzero].mean(axis=0)
            ) / (feat_std[nonzero] + 1e-8)
            meta[:, 2:12] = pca_feats

    return meta


def load_waveform_data():
    """Load Seattle Children's waveform tables and return a record_id keyed lookup."""
    base_dir = os.path.abspath(os.path.join(script_dir, "../../data/db_tables/sch/waveforms"))
    waveform_frames = []
    waveform_cols = ["record_id"] + [f"value_{i}" for i in range(256)]

    for filename in sorted(os.listdir(base_dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(base_dir, filename))
            missing_cols = [col for col in waveform_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Waveform file {filename} is missing expected columns: {missing_cols[:5]}"
                )
            waveform_frames.append(df[waveform_cols])

    if not waveform_frames:
        raise FileNotFoundError(f"No waveform CSVs found under {base_dir}")

    waveform_df = pd.concat(waveform_frames, ignore_index=True)
    waveform_df = waveform_df.drop_duplicates(subset="record_id").set_index("record_id")

    # New dataset layout stores waveforms directly by record_id, so this is an identity map.
    record_to_bank = {record_id: record_id for record_id in waveform_df.index}

    return waveform_df, record_to_bank


def get_waveform_by_record_id(record_id, waveform_df, record_to_bank):
    """Get waveform values for a record_id. Returns None if not found or has NaN."""
    waveform_id = record_to_bank.get(record_id)
    if waveform_id is None or waveform_id not in waveform_df.index:
        return None
    wf = waveform_df.loc[waveform_id].values.astype(np.float32)
    # Check for NaN in raw waveform
    if np.isnan(wf).any():
        return None
    return wf


class WaveformSequenceDataset:
    """
    Prepares sequences with (wave, meta, y) from merged dataframes.
    
    wave: [T, 1, L] raw waveforms
    meta: [T, M] non-waveform features (freq one-hot + last_level + PCA features)
    y: [T] labels
    
    HYBRID MODE: includes PCA features in meta for stability.
    """
    
    def __init__(self, merged_dfs, mapping, waveform_df, record_to_bank, device='cpu', include_pca=True):
        self.sequences = []
        self.raw_sequences = []
        self.device = device
        self.include_pca = include_pca
        
        for df in merged_dfs:
            seq_data = self._process_df(df, mapping, waveform_df, record_to_bank)
            if seq_data is not None:
                raw_seq, norm_seq = seq_data
                self.raw_sequences.append(raw_seq)
                self.sequences.append(norm_seq)
    
    def _process_df(self, df, mapping, waveform_df, record_to_bank):
        """Convert a single sequence dataframe to (wave, meta, y) tensors."""
        record_ids = df['record_id'].values
        
        # Collect waveforms
        waveforms = []
        for rid in record_ids:
            wf = get_waveform_by_record_id(rid, waveform_df, record_to_bank)
            if wf is None:
                return None  # Skip sequence if any waveform is missing
            waveforms.append(wf)
        
        waveforms = np.stack(waveforms, axis=0)  # [T, L]
        
        if self.include_pca:
            # HYBRID: freq (1) + last_level (1) + extracted features (10) = 12 dims
            raw_meta = df.iloc[:, 1:13].values.astype(np.float32)  # [T, 12]
        else:
            # Original: freq (1) + last_level (1) = 2 dims
            raw_meta = df.iloc[:, 1:3].values.astype(np.float32)  # [T, 2]

        meta = normalize_meta(raw_meta, include_pca=self.include_pca)
        
        # Labels — label columns start at index 13 (after input_df features)
        labels = df.iloc[:, 13:].idxmax(axis=1).map(mapping).values
        
        # Convert to tensors
        wave = torch.tensor(waveforms, dtype=torch.float32, device=self.device)
        wave = wave.unsqueeze(1)  # [T, 1, L]
        meta = torch.tensor(meta, dtype=torch.float32, device=self.device)
        raw_meta = torch.tensor(raw_meta, dtype=torch.float32, device=self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        raw_seq = (wave, raw_meta, y)
        norm_seq = (wave, meta, y)
        return raw_seq, norm_seq
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __iter__(self):
        return iter(self.sequences)


def split_for_testing_waveform(sequences):
    """
    Split sequences into prefix records for testing.
    Returns list of (wave_prefix, meta_prefix, y_prefix) tuples.
    """
    records = []
    raw_sequences = getattr(sequences, "raw_sequences", sequences)
    for wave, raw_meta, y in raw_sequences:
        for i in range(len(wave)):
            wave_prefix = wave[:i+1]
            meta_prefix = raw_meta[:i+1]
            label_prefix = y[:i+1]

            meta_prefix_np = meta_prefix.detach().cpu().numpy()
            normalized_prefix = normalize_meta(
                meta_prefix_np,
                include_pca=(meta_prefix_np.shape[1] >= 12),
            )
            normalized_prefix = torch.tensor(
                normalized_prefix,
                dtype=meta_prefix.dtype,
                device=meta_prefix.device,
            )

            records.append((wave_prefix, normalized_prefix, label_prefix))
    return records
