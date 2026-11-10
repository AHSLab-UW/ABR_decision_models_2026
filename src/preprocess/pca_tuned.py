import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.common.data_util import *
import joblib


def apply_pca(df, waveform_cols, n_components, pca_model):
    non_waveform_cols = [c for c in df.columns if c not in waveform_cols]
    X = df[waveform_cols].to_numpy()
    pca_extraction = pca_model.transform(X)
    extracted_df = pd.DataFrame(
        pca_extraction,
        columns=[f'extracted_{i + 1}' for i in range(n_components)],
        index=df.index
    )

    waveform_start = df.columns.get_loc(waveform_cols[0])
    before = [c for c in df.columns[:waveform_start] if c not in waveform_cols]
    after = [c for c in df.columns[waveform_start:] if c not in waveform_cols]
    return pd.concat([df[before], extracted_df, df[after]], axis=1)

def extract_features_train_test(train_df, test_df, waveform_cols, n_components=10, pca_model=None):
    if pca_model is None:
        X_train = train_df[waveform_cols].to_numpy()
        pca_model = PCA(n_components=n_components, whiten=True)
        pca_model.fit(X_train)
        #print(f"Shape: {X_train.shape}")
        #print(f"Rank: {np.linalg.matrix_rank(X_train)}")
        #print(f"Per-feature variance: {X_train.var(axis=0)}")
        #print(f"Any NaN: {np.isnan(X_train).any()}")
        #print(f"Any constant columns: {(X_train.var(axis=0) == 0).sum()}")
        #print(f"Total Variance Captured by {n_components} PCs: "
        #      f"{sum(pca_model.explained_variance_ratio_) * 100:.2f}%")


    test_out, train_out = None, None

    if train_df is not None:
        train_out = apply_pca(train_df, waveform_cols, n_components, pca_model)
    if test_df is not None:
        test_out = apply_pca(test_df, waveform_cols, n_components, pca_model)

    return train_out, test_out, pca_model


"""dfs = load_dfs("../../data/feature_tables/sch/full_table")

feature_df = pd.concat(dfs, ignore_index=True)

print("Final shape:", feature_df.shape)
waveform_cols = [col for col in feature_df.columns if col not in ['record_id','freq','last_level']]
final_input_df, pca_model = extract_features(feature_df, waveform_cols, n_components=10)
# TODO this joblib dump only works for python native exports
# need to look into dumping raw SVD decomposition
# or we pivot to giving Rafael a bundled python dll
joblib.dump(pca_model, '../../exported_features/stimulus_change/preprocess/pca_model.joblib')
final_input_df.to_csv('../../data/feature_tables/sch/input.csv', index=False)
print("Saved PCA features")
"""