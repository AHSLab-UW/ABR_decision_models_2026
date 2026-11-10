import glob
import json
import os
import math

import numpy as np
import pandas as pd
import re
from datetime import datetime

script_dir = os.path.dirname(os.path.realpath(__file__))

# Bio-logic exports start each waveform 1 ms before stimulus onset, so sample i sits at
# i * epoch / points - 1 ms; the clinician-marked latencies are relative to stimulus onset.
WAVEFORM_START_MS = -1.0

def waveform_times(epoch, n_samples):
    return np.arange(n_samples) * epoch / n_samples + WAVEFORM_START_MS

def load_resource(name):
    csv_path = os.path.join(script_dir, f"../../data/{name}")
    return pd.read_csv(csv_path)

def get_dataset(namespace):
    record_df = load_resource(f"db_tables/{namespace}/records_table.csv")
    session_df = load_resource(f"db_tables/{namespace}/sessions_table.csv")
    waveform_dfs = load_dfs(f"{script_dir}/../../data/db_tables/{namespace}/waveforms")
    waveform_df = pd.concat(waveform_dfs, ignore_index=True)
    return session_df, record_df, waveform_df


def filter_out_ones(x):
  if (len(x) <= 1):
    return False
  return True

def get_grouped_dfs(record_df, grouped_columns, output=False, save_folder=None, sort=False, filter_func=None, mapper_func=None):

  grouped_by_session = record_df.groupby(grouped_columns)

  dataframes = []

  for (grouped_columns),group in grouped_by_session:
    df = group.reset_index(drop=True)
    if sort and 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        df = df.sort_values(by='start_time', ascending=True)
    dataframes.append(df)
    if output:
      df.to_csv(f"../../data/split_tests/{save_folder}/{'_'.join(map(lambda value: str(value), grouped_columns))}.csv", index=False)

  if mapper_func is not None:
      dataframes = list(map(mapper_func, dataframes))

  if filter_func is not None:
      dataframes = list(filter(filter_func, dataframes))

  return dataframes

def get_value(string):
  string = string[1:-1]
  if not (string or string.strip()):
    return None
  return string


def parse_date(date_str):
  formatter = "%m/%d/%Y %I:%M:%S %p"
  date_obj = datetime.strptime(date_str, formatter)
  return date_obj


def get_middle_value(freq, value):
  result = re.split(r'\s+', freq)

  if len(result) < value:
    return None

  return result[value]


def load_dfs(path):
    dfs = []
    for files in os.walk(path):
        for file in files[2]:
            df = pd.read_csv(path + "/" + file)
            dfs.append(df)
    return dfs

def clean_files(path):
    files = glob.glob(os.path.join(path, '*'))

    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted: {file}")

def split_df_across_files(df, lines_per_file, output_prefix):
    num_files = math.ceil(len(df) / lines_per_file)

    for i in range(num_files):
        start = i * lines_per_file
        end = (i + 1) * lines_per_file

        chunk = df.iloc[start:end]

        filename = f"{output_prefix}_{i:03d}.csv"
        chunk.to_csv(filename, index=False)

        print(f"Wrote {filename} ({len(chunk)} rows)")

def file_exists(path):
    return os.path.exists(path)

def get_waveform(record_id, waveform_df):
    values = waveform_df[waveform_df['record_id'] == record_id].iloc[0, 1:].to_numpy(dtype=float)
    return values[~np.isnan(values)].tolist()

def read_json_dict(path):
    with open(path, 'r') as f:
        data_dict = json.load(f)
    return data_dict