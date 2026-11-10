import sys

import numpy as np

from src.common.model.stimulus_change_model_util import best_config_lstm
from src.common.model.wave_5_model_util import *
from src.models.wave_5.lstm.wave_5_lstm_model import Wave5LSTMModel
from src.models.stimulus_change.lstm.configs.lstm_configurations import model_configs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

print(f"Initializing data pipeline...")
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/wave_5_action_table.csv")
input_data = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/input.csv")

merged_dfs, big_merged_df, label_ids = get_seq_split_inputs_outputs(action_df, input_data)

mapping = {True: 1, False: 0}
reverse_mapping = {1: True, 0: False}
labels_categorical = np.array(label_ids.map(mapping))
labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]

train_dfs, test_dfs, holdout_dfs = test_train_holdout_split_list(merged_dfs)
scale_dfs_2(train_dfs, test_dfs, holdout_dfs)

train_split = prepare_sequences_wave5(train_dfs, mapping)
test_split = prepare_sequences_wave5(test_dfs, mapping)
holdout_split = prepare_sequences_wave5(holdout_dfs, mapping)

evaluate_grid_search(
    model_configs=best_config_lstm,
    model_factory=Wave5LSTMModel,
    train_split=train_split,
    test_split=test_split,
    holdout_split=holdout_split,
    unique_labels=labels,
    labels_categorical=labels_categorical,
    sequential=True,
    root_dir=ROOT_DIR,
    namespace="lstm"
)