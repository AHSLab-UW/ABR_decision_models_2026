import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.common.model.wave_5_model_util import *
from src.models.wave_5.dnn.wave_5_dnn_model import Wave5DNNModel
from src.models.stimulus_change.dnn.configs.dnn_configurations import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

print(f"Initializing data pipeline...")
action_df = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/wave_5_action_table.csv")
input_data = pd.read_csv(f"{ROOT_DIR}/data/feature_tables/{train_dataset_namespace}/input.csv")

action_df = action_df.drop(['stop'], axis=1) # dont consider sequences here so no need to split
input_df, big_merged_df, label_ids = get_inputs_outputs(input_data, action_df)
mapping = {True: 1, False: 0}
reverse_mapping = {1: True, 0: False}
labels_categorical = np.array(label_ids.map(mapping))
labels = [reverse_mapping[i] for i in range(len(reverse_mapping))]
#scale_dfs([big_merged_df])
train_df, test_df, holdout_df = test_train_holdout_split_df(big_merged_df)
scale_dfs_2([train_df], [test_df], [holdout_df])

train_data, train_labels = split_df_data_labels(train_df, mapping)
test_data, test_labels = split_df_data_labels(test_df, mapping)
holdout_data, holdout_labels = split_df_data_labels(holdout_df, mapping)

train_batches = create_batches(train_data, train_labels)
test_batches = create_batches(test_data, test_labels)
holdout_batches = create_batches(holdout_data, holdout_labels)

evaluate_grid_search(
    model_configs=[best_config_dnn],
    model_factory=Wave5DNNModel,
    train_split=train_batches,
    test_split=test_batches,
    holdout_split=holdout_batches,
    unique_labels=labels,
    labels_categorical=labels_categorical,
    sequential=False,
    root_dir=ROOT_DIR,
    namespace="dnn"
)
#evaluate_grid_search([cross_val_config_dnn], epochs, learning_rates, [])
#evaluate_grid_search([best_config_dnn], epochs, learning_rates, [])