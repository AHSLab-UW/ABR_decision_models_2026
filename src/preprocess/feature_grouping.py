import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample

from src.common.data_util import *

target_freqs = np.log2([500, 1000, 2000, 4000, 8000])

def group_features(feature_df, record_df, waveform_df, output):
    feature_df['freq'] = feature_df['freq'].map(lambda x: math.log2(x))

    feature_df = feature_df[['record_id','freq', 'last_level']]
    waveform_rows = []
    missing_records = []
    short_epoch_records = []

    for idx, row in feature_df.iterrows():
        record_id = row['record_id']

        waveform_row = waveform_df.loc[waveform_df['record_id'] == record_id]
        if waveform_row.empty:
            continue

        if not np.isclose(target_freqs, row['freq']).any():
            continue

        record = record_df.loc[record_df['record_id'] == record_id]

        epoch = record['epoch'].iloc[0]
        waveform_values = waveform_row.iloc[0, 1:257].values.astype(float)

        if np.isnan(waveform_values).any():
            continue

        target_downsample_freq = 4000

        #assuming we are in ms we can assume target_downsample_freq/1000 datapoints per milisecond
        # so for 4 kHz we have 4 dps per ms
        time_window_max = 15
        time_window_min = 4

        # recording too short to contain the full window (the waveform starts at WAVEFORM_START_MS)
        if epoch + WAVEFORM_START_MS < time_window_max:
            short_epoch_records.append(record_id)
            continue

        # time windows are in ms after stimulus onset; shift by the waveform start, then divide by
        # the time per sample to get sample indices (4–15 ms post-stimulus = samples at 5–16 ms)
        time_per_waveform = epoch / 256
        five_ms_index = int((time_window_min - WAVEFORM_START_MS) / time_per_waveform)
        fifteen_ms_index = int((time_window_max - WAVEFORM_START_MS) / time_per_waveform)

        sliced_wave = waveform_values[five_ms_index:fifteen_ms_index + 1]

        # TODO look into how to downsample to non-discrete windows of time
        # ie 2500 kHz -> 2.5 dps per second? no guarantee that we have discrete pts
        resampled_window_2 = resample(sliced_wave, (time_window_max - time_window_min) * int((target_downsample_freq / 1000)))
        waveform_data = {
            f'last_data_point_{i}': val for i, val in enumerate(resampled_window_2)
        }
        waveform_data['record_id'] = record_id
        waveform_rows.append(waveform_data)
        if idx % 10000 == 0:
            print(f"Processed {((idx/len(feature_df))*100):.2f}% of waveforms")

    print(f"Skipped {len(short_epoch_records)} records too short to reach 15 ms post-stimulus")
    waveform_df_expanded = pd.DataFrame(waveform_rows)
    feature_df_final = pd.merge(feature_df, waveform_df_expanded, on='record_id', how='inner')
    clean_files(f"{output}/full_table")
    split_df_across_files(feature_df_final, 50000, f"{output}/full_table/full_feature_table")
    return feature_df_final

