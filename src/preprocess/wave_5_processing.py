import pandas as pd

def process_sequence(seq_df):
    df = seq_df[seq_df['averages'] >= 750].copy()

    if df.empty:
        return None
    df['amplitude_5'] = df.groupby('db')['amplitude_5'].transform(lambda x: x.ffill().bfill())
    df['latency_5'] = df.groupby('db')['latency_5'].transform(lambda x: x.ffill().bfill())

    df = df.sort_values(by='db', ascending=True)
    has_explicit_marker = df['amplitude_5'].notna()
    df['has_wave_5'] = has_explicit_marker.cummax().astype(bool)
    df.rename(columns={'id': 'record_id'}, inplace=True)

    return df[['record_id', 'has_wave_5']]


def build_wave_5_action_table(sequences, action_df, output_path):
    wave_5_records = []

    print("Processing sequences...")
    for i, seq_df in enumerate(sequences):
        processed_seq = process_sequence(seq_df)

        if processed_seq is not None:
            wave_5_records.append(processed_seq)

        if (i + 1) % 500 == 0:
            print(f"Processed {((i + 1) / len(sequences) * 100):.2f}% of sequences")

    if not wave_5_records:
        print("No valid records found.")
        return

    all_wave_5_df = pd.concat(wave_5_records, ignore_index=True)

    final_action_df = action_df[['record_id', 'stop']].copy()


    final_action_df = pd.merge(final_action_df, all_wave_5_df, on='record_id', how='left')

    final_action_df['has_wave_5'] = final_action_df['has_wave_5'].fillna(False)

    final_action_df = final_action_df[['has_wave_5', 'record_id', 'stop']]

    final_action_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved new feature table to: {output_path}")

    wave_5_on_stop = ((final_action_df['has_wave_5'] == True) & (final_action_df['stop'] == 1)).sum()
    total_stops = (final_action_df['stop'] == 1).sum()
    print(f"Number of sequences with wave 5 on a stop: {wave_5_on_stop}/{total_stops}")
    print(f"Total positive Wave 5 labels generated: {(final_action_df['has_wave_5'] == True).sum()}")
    return final_action_df

"""
print(f"Initializing pure Pandas data pipeline...")

action_df = pd.read_csv("../../data/feature_tables/sch/action_table.csv")
input_df = pd.read_csv("../../data/feature_tables/sch/input.csv")

sequences, _, _ = get_seq_split_inputs_outputs(action_df, input_df)

build_wave_5_action_table(sequences, action_df, banks_df, record_df, "../../data/feature_tables/sch/wave_5_action_table.csv")
"""