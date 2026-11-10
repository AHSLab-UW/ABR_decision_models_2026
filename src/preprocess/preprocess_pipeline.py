import joblib

from src.preprocess.action_table import *
from src.preprocess.feature_grouping import group_features

def preprocess_dataset(namespace, output):
    session_df, record_df, waveform_df = get_dataset(namespace)
    print("Building Action Tables")
    action_df, incomplete_feature_df, _ = build_action_tables(record_df, output)
    print("Downsampling features")
    group_features(incomplete_feature_df, record_df, waveform_df, output)

preprocess_dataset('sch', '../../data/feature_tables/sch')
preprocess_dataset("uw", "../../data/feature_tables/uw")

