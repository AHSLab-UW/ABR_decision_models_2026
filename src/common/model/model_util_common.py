import joblib
import numpy as np
import torch
import random
from sklearn.preprocessing import StandardScaler

from src.common.data_util import *

def is_higher_than_saved_acc(accuracy, saved_results_path):
    if not file_exists(saved_results_path):
        return True
    with open(saved_results_path) as f:
        results = json.load(f)
        prev_acc = results["accuracy"]
        return accuracy > prev_acc

def prepare_sequences(dfs, label_map, class_index=13):
    splits = []
    for df in dfs:
        df_labels = df.iloc[:, class_index:].idxmax(axis=1)
        df_labels = df_labels.map(label_map).to_numpy()
        x = torch.tensor(df.iloc[:, 1:class_index].values, dtype=torch.float32)
        y = torch.tensor(df_labels, dtype=torch.int64)
        splits.append((x, y))
    return splits

def split_indiv_records(lst):
    records = []
    for seq, label in lst:
        total_len = len(seq)
        for i in range(total_len):
            temp_seq = seq[:i+1, :]
            label_seq = label[:i+1]
            records.append((temp_seq, label_seq, total_len))
    return records

def flatten_labels(nested_list):
    result = []
    for x in nested_list:
        if isinstance(x, (list, tuple, np.ndarray)):
            result.extend(flatten_labels(x))
        else:
            result.append(x)
    return result

def split_action_table(action_df):
    split_indices = action_df.index[action_df['stop'] == 1].tolist()

    prev = 0
    dfs = []
    for index in split_indices:
        dfs.append(action_df.iloc[prev:index + 1, :])
        prev = index+1
    return dfs

def undersample_class(df, class_index, ratio):
    target_col_idx = 13 + class_index

    mask_target = df.iloc[:, target_col_idx] == 1

    df_target = df[mask_target]
    df_other = df[~mask_target]

    n_other = len(df_other)
    n_current_target = len(df_target)

    n_keep = int((ratio * n_other) / (1 - ratio))

    print(f"Undersampling Class {class_index}: Reducing {n_current_target} -> {n_keep} rows "
          f"to hit {ratio * 100:.1f}% distribution.")

    df_target_sampled = df_target.sample(n=n_keep)

    result = pd.concat([df_other, df_target_sampled])
    result_df = result.sample(frac=1).reset_index(drop=True)

    print_df_label_distribution(result_df)

    return result_df

def print_df_label_distribution(result_df):
    total_rows = len(result_df)

    label_cols = result_df.columns[13:]

    for i, col_name in enumerate(label_cols):
        count = result_df[col_name].sum()
        pct = (count / total_rows) * 100

        print(f"Class {i} ({col_name}): {count}/{total_rows} ({pct:.2f}%)")

def oversample_minorities(df, min_count=2000):
    label_cols = df.columns[13:]

    generated_rows = []

    for i, col_name in enumerate(label_cols):
        class_mask = df[col_name] == 1
        class_df = df[class_mask]
        current_count = len(class_df)

        if current_count == 0:
            continue

        if current_count < min_count:
            needed = min_count - current_count
            print(f"Class {i} ({col_name}): Increasing {current_count} -> {min_count} rows (+{needed})")

            oversampled_batch = class_df.sample(n=needed, replace=True)
            generated_rows.append(oversampled_batch)

    if generated_rows:
        df_augmented = pd.concat([df] + generated_rows)
        print("After oversampling: ")
        print_df_label_distribution(df_augmented)
        return df_augmented.sample(frac=1).reset_index(drop=True)
    else:
        print("No classes needed oversampling.")
        return df

# assumes validate-set is just remaining dfs
def test_train_holdout_split_list(df_list, train=0.8, test=0.1, holdout=0.1):
    if abs((train + test + holdout) - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0. Current sum: {train + test + holdout}")

    session_groups = {}
    for df in df_list:
        session_id = df['session_id'].iloc[0]
        session_groups.setdefault(session_id, []).append(df)

    session_ids = list(session_groups.keys())
    random.shuffle(session_ids)

    n = len(session_ids)
    train_end = int(n * train)
    test_end = train_end + int(n * test)

    train_ids = session_ids[:train_end]
    test_ids = session_ids[train_end:test_end]
    holdout_ids = session_ids[test_end:]

    train_dfs = [df for sid in train_ids for df in session_groups[sid]]
    test_dfs = [df for sid in test_ids for df in session_groups[sid]]
    validate_dfs = [df for sid in holdout_ids for df in session_groups[sid]]

    # session_id no longer meaningful once split is resolved
    train_dfs = [df.drop(columns=['session_id']) for df in train_dfs]
    test_dfs = [df.drop(columns=['session_id']) for df in test_dfs]
    validate_dfs = [df.drop(columns=['session_id']) for df in validate_dfs]

    return train_dfs, test_dfs, validate_dfs

def test_train_holdout_split_df(df, train=0.8, test=0.1, holdout=0.1):
    if abs((train + test + holdout) - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0. Current sum: {train + test + holdout}")

    session_ids = df['session_id'].unique().tolist()
    random.shuffle(session_ids)

    n = len(session_ids)
    train_end = int(n * train)
    test_end = train_end + int(n * test)

    train_ids = session_ids[:train_end]
    test_ids = session_ids[train_end:test_end]
    holdout_ids = session_ids[test_end:]

    train_df = df[df['session_id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['session_id'].isin(test_ids)].reset_index(drop=True)
    validate_df = df[df['session_id'].isin(holdout_ids)].reset_index(drop=True)

    # session_id no longer meaningful once split is resolved
    train_df = train_df.drop(columns=['session_id'])
    test_df = test_df.drop(columns=['session_id'])
    validate_df = validate_df.drop(columns=['session_id'])

    return train_df, test_df, validate_df
def scale_dfs(merged_dfs):
    for df in merged_dfs:
        df.iloc[:, 3:13] = StandardScaler().fit_transform(df.iloc[:, 3:13])
        df.iloc[:, [2]] = StandardScaler().fit_transform(df.iloc[:, [2]])

def scale_dfs_2(train_dfs, test_dfs, holdout_dfs):
    # PCA already does scaling on extracted features, just need to scale dB
    feature_cols = train_dfs[0].columns[[2]]
    scaler = StandardScaler()

    flat_train_df = pd.concat(train_dfs, ignore_index=True)
    scaler.fit(flat_train_df[feature_cols])

    for df in train_dfs:
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])
    for df in test_dfs:
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])
    for df in holdout_dfs:
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

    return scaler


def scale_dfs_causal(merged_dfs):
    for df in merged_dfs:
        raw_pca = df.iloc[:, 3:13].values.astype(np.float64)
        raw_level = df.iloc[:, 2].values.astype(np.float64)

        T = len(df)

        scaled_pca = raw_pca.copy()
        scaled_level = raw_level.copy()

        eps = 1e-8

        level_sum = 0.0
        level_sumsq = 0.0

        pca_cols = raw_pca.shape[1]
        pca_sum = np.zeros(pca_cols, dtype=np.float64)
        pca_sumsq = np.zeros(pca_cols, dtype=np.float64)

        for t in range(T):
            count = t + 1

            val_level = raw_level[t]
            level_sum += val_level
            level_sumsq += val_level * val_level

            level_mean = level_sum / count
            level_var = max((level_sumsq / count) - (level_mean * level_mean), 0.0)
            level_std = np.sqrt(level_var)

            if level_std > 0:
                scaled_level[t] = (val_level - level_mean) / (level_std + eps)

            val_pca = raw_pca[t]
            pca_sum += val_pca
            pca_sumsq += val_pca * val_pca

            pca_mean = pca_sum / count
            pca_var = np.maximum((pca_sumsq / count) - (pca_mean * pca_mean), 0.0)
            pca_std = np.sqrt(pca_var)

            nonzero = pca_std > 0
            if nonzero.any():
                scaled_pca[t, nonzero] = (val_pca[nonzero] - pca_mean[nonzero]) / (pca_std[nonzero] + eps)

        df.iloc[:, 3:13] = scaled_pca.astype(np.float32)
        df.iloc[:, 2] = scaled_level.astype(np.float32)

    return merged_dfs

def scale_dfs_anchored(train_dfs, test_dfs, holdout_dfs):
    feature_cols = train_dfs[0].columns[1:13]

    def apply_anchor(dfs):
        for df in dfs:
            anchor_values = df.iloc[0][feature_cols].copy()

            df.loc[:, feature_cols] = df[feature_cols] - anchor_values

    apply_anchor(train_dfs)
    apply_anchor(test_dfs)
    apply_anchor(holdout_dfs)

    scaler = StandardScaler()

    flat_train = pd.concat(train_dfs, ignore_index=True)
    scaler.fit(flat_train[feature_cols])

    for df in train_dfs:
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])
    for df in test_dfs:
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])
    for df in holdout_dfs:
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

def scale_dfs_lstm_final(train_dfs, test_dfs, holdout_dfs):
    base_features = list(train_dfs[0].columns[2:13])

    def append_deltas(dfs):
        for df in dfs:
            for i, col in enumerate(base_features):
                df.insert(13 + i, f"{col}_delta", df[col].diff().fillna(0.0))

    append_deltas(train_dfs)
    append_deltas(test_dfs)
    append_deltas(holdout_dfs)

    all_features = base_features + [f"{c}_delta" for c in base_features]
    scaler = StandardScaler()
    flat_train = pd.concat(train_dfs, ignore_index=True)
    scaler.fit(flat_train[all_features])

    for df in train_dfs + test_dfs + holdout_dfs:
        df.loc[:, all_features] = scaler.transform(df[all_features]).astype(np.float32)

def create_batches(x_tensor, y_tensor, batch_size=64, shuffle=True):
    if shuffle:
        indices = torch.randperm(x_tensor.size(0))
        x_tensor = x_tensor[indices]
        y_tensor = y_tensor[indices]

    batches = []
    for i in range(0, len(x_tensor), batch_size):
        x_batch = x_tensor[i: i + batch_size]
        y_batch = y_tensor[i: i + batch_size]
        batches.append((x_batch, y_batch))

    return batches

def print_model_architecture(model, config):
    config = json.dumps(config, indent=4)
    print("Model configuration:\n")
    print(config)

    total_params = 0
    for name, p in model.named_parameters():
        total = p.numel()
        total_params += total

    print("\nTOTAL PARAMETERS:", total_params)


def eval_val_accuracy(model, val_set):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for seq, labels in val_set:
            if torch.isnan(seq).any() or torch.isinf(seq).any():
                continue

            preds, _ = model.forward_test(seq)
            preds = preds.view(-1)
            labels = labels.view(-1)

            preds = preds.long()
            labels = labels.long()

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return (correct / total * 100) if total > 0 else 0

train_dataset_namespace = "sch"