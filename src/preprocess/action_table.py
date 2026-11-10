import os
import pandas as pd
import numpy as np
from src.common.plot_util import create_histogram
from src.common.data_util import *


def create_matrix_df(grouped_dfs, temp_file, db_changes):
    columns = ["<-30", "-30", "-25", "-20", "-15", "-10", "-5", "0", "5", "10", "15", "20", ">20", "stop",
               "record_id", "session_id"]
    rows = []
    db_to_record_id = {col: [] for col in columns}

    for df in grouped_dfs:
        db_diff = df['db'].shift(-1) - df['db']
        db_vals = df['db'].values
        id_vals = df['record_id'].values
        session_id_vals = df['session_id'].values
        db_diff_vals = db_diff.values

        freq_vals = df['freq'].values if temp_file else None

        for idx in range(len(df)):
            record_id = id_vals[idx]
            diff = db_diff_vals[idx]
            session_id = session_id_vals[idx]

            row = {col: 0 for col in columns}
            row['record_id'] = record_id
            row['session_id'] = session_id

            if temp_file:
                row['freq'] = freq_vals[idx]
                row['last_level'] = db_vals[idx]
            else:
                category = None
                if pd.isna(diff):
                    category = "stop"
                else:
                    db_changes.append(diff)
                    rounded = int(round(diff / 5.0) * 5)

                    if rounded > 20:
                        category = ">20"
                    elif rounded < -30:
                        category = "<-30"
                    else:
                        category = str(rounded)

                if category:
                    row[category] = 1
                    if category in db_to_record_id:
                        db_to_record_id[category].append(record_id)

            rows.append(row)

    matrix_df = pd.DataFrame(rows)

    cols_to_keep = ['record_id', 'session_id']
    if temp_file:
        cols_to_keep.extend(['freq', 'last_level'])

    for col in columns:
        if col in matrix_df.columns and matrix_df[col].sum() > 0 and col not in cols_to_keep:
            cols_to_keep.append(col)

    return matrix_df[cols_to_keep]

def build_action_tables(record_df, output_dir):
    print(f"Grouping DataFrames...")
    grouped_dfs = get_grouped_dfs(
        record_df,
        ['session_id', 'freq', 'ear'],
        sort=True,
        output=False,
        save_folder=output_dir,
        filter_func=filter_out_ones
    )
    first_db_lvls = list(map(lambda x: x.loc[0, 'db'], grouped_dfs))
    create_histogram(first_db_lvls, 5, 'First DB Level in Sequence', 'Occurences', 'First DB Level in Sequence')

    db_changes = []
    print(f"Number of sequences: {len(grouped_dfs)}")

    action_df = create_matrix_df(grouped_dfs, False, db_changes)
    os.makedirs(output_dir, exist_ok=True)
    action_df.to_csv(f"{output_dir}/action_table.csv", index=False)

    create_histogram(db_changes, 5, 'Δ dB', 'Number of Occurrences',
                     f'Distribution of dB Jumps')

    incomplete_feature_df = create_matrix_df(grouped_dfs, True, db_changes)
    #incomplete_feature_df.iloc[:, -3:].to_csv(f"{output_dir}/incomplete_feature_table.csv", index=False)

    return action_df, incomplete_feature_df