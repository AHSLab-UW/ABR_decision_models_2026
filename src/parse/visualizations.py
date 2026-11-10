import random
from venv import create

from matplotlib.colors import SymLogNorm
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import confusion_matrix

from src.common.plot_util import *
from src.common.data_util import *

# hardcode to sch dataset for now

def plot_averages(record_df, save_path):
    record_counts = record_df['averages'].tolist()
    create_histogram(record_counts, 100, 'Number of Averages', 'Number of Occurrences', 'Distribution of Waveform Averages', save_path=save_path + "/dataset_averages.png", x_bound=(0, 4000))
    """
    bins = list(range(0, 4000 + 1, 100))

    plt.hist(record_counts, bins=bins, edgecolor='black')

    tick_spacing = 500
    xticks = list(range(0, 4000 + 1, tick_spacing))
    plt.xticks(xticks, rotation=45)
    plt.xlabel('Number of Averages')
    plt.ylabel('Number of Occurrences')
    plt.title('Distribution of Waveform Averages')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path + "/dataset_averages.png", dpi=300, bbox_inches='tight', pad_inches=0.1)"""

def plot_session_lengths(record_df, save_path):
    session_lengths = record_df.groupby('session_id').size().tolist()
    create_histogram(session_lengths, 5, 'Waveforms Per Session', 'Occurrences', 'Waveforms Per Session', save_path=save_path + "/dataset_session_lengths.png")

def plot_session_lengths_by_freq(record_df, save_path):
    frequencies = [500, 1000, 2000, 4000, 8000]

    for freq in frequencies:
        filtered_df = record_df[record_df['freq'] == freq]
        session_lengths = filtered_df.groupby('session_id').size().tolist()
        create_histogram(session_lengths, 5, 'Waveforms Per Session', 'Number of Sessions', f"Waveforms Per Session for {freq} Hz", save_path=f"{save_path}/dataset_session_lengths_{freq}.png")


def plot_bone_ear_distrib(record_df, save_path):
    sessions = 0
    both = 0
    left = 0
    right = 0
    duration_list = []
    bone_duration = []
    normal_duration = []

    for session_id, group in record_df.groupby('session_id'):

        sessions += 1
        has_left = (group['ear'] == 'Left').any()
        has_right = (group['ear'] == 'Right').any()
        if has_left and has_right:
            both += 1
        elif has_left and not has_right:
            left += 1
        elif has_right and not has_left:
            right += 1

        start_times = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), group['start_time'].tolist()))
        start_times.sort()
        start = start_times[0]
        end = start_times[-1]
        duration = end - start
        minutes = duration.seconds / 60
        duration_list.append(minutes)
        if 'Bone Oscillator' in group['transducer'].values:
            bone_duration.append(minutes)
        else:
            normal_duration.append(minutes)

    print(f"Both Ears: {both}/{sessions} Sessions")
    print(f"Left Only: {left}/{sessions} Sessions")
    print(f"Right Only: {right}/{sessions} Sessions")

    # temp hotfix to fix weird graph rendering
    # extreme outliers in sch dataset
    x_bounds = None
    if "sch" in save_path:
        x_bounds = (0, 200)

    create_histogram(duration_list, 5, 'Minutes','Number of Sessions', 'Duration of Sessions', save_path=save_path + "/dataset_session_duration.png", x_bound=x_bounds)
    create_histogram(bone_duration, 5, 'Minutes', 'Number of Sessions', 'Duration of Session with Bone Conduction', save_path=save_path + "/dataset_session_bone_duration.png", x_bound=x_bounds)
    create_histogram(normal_duration, 5, 'Minutes', 'Number of Sessions', 'Duration of Session with Air Conduction', save_path=save_path + "/dataset_session_air_duration.png", x_bound=x_bounds)

def plot_db_freq_distribution(record_df, save_path):
    record_counts = record_df['db']

    create_histogram(record_counts.tolist(), 5, 'Waveform dB', 'Occurrences', 'Distribution of dB Level', save_path=save_path + "/dataset_db_distribution.png")

    record_counts = record_df['freq']

    unique_vals, counts = np.unique(record_counts.to_numpy(), return_counts=True)

    included_freqs =  np.isin(unique_vals, [500, 1000, 2000, 4000, 8000])
    unique_vals = unique_vals[included_freqs]
    counts = counts[included_freqs]

    create_bar_chart(dict(zip(unique_vals, counts)),  'freq', 'Occurrences', 'Distribution of freq Level', save_path=save_path + "/dataset_freq_distribution.png")



def prepare_waterfall_plot_data(sequences_to_plot, record_df, waveform_df):
    dfs = load_dfs("../../data/cross_validation_sets/stimulus_change/lstm")
    random.shuffle(dfs)
    dfs = dfs[:sequences_to_plot]
    record_ids = list(map(lambda df: df.iloc[:, 0], dfs)) # extract record id from these
    sequences = []
    for df in record_ids:
        seq_entry = []
        for i in range(len(df)):
            record_id = df.iloc[i]
            record_entry = record_df.loc[record_df['id'] == record_id]
            epoch = record_entry['epoch'].iloc[0]
            db = record_entry['db'].iloc[0]
            amplitude_5 = record_entry['amplitude_5'].iloc[0]
            latency_5 = record_entry['latency_5'].iloc[0]
            waveform_entry = waveform_df.loc[waveform_df['record_id'] == record_id]
            waveform_pts = waveform_entry.iloc[0, 1:].tolist()
            seq_entry.append((waveform_pts, epoch, amplitude_5, latency_5, db, record_id))
        sequences.append(seq_entry)
    return sequences

def plot_waterfalls(record_df):
    data = prepare_waterfall_plot_data(20, record_df)
    for i in range(len(data)):
     print(f"Printing Plot {i+1}")
     waterfall_plot_sequence_non_sequential(data[i], start_time=-1)


def plot_sequence_length(record_df, save_path):
    grouped_dfs = get_grouped_dfs(record_df, ['session_id', 'freq', 'ear'], sort=True,
                                   output=False, save_folder="seattle_childrens",
                                   filter_func=filter_out_ones)
    sequence_length = list(map(lambda x: len(x), grouped_dfs))
    first_db_in_seq = list(map(lambda x: x['db'].iloc[0], grouped_dfs))

    create_histogram(sequence_length, 2, 'Number of Waveforms', 'Number of Sequences',
                     "Waveforms per Sequence",
                     save_path=save_path + "/dataset_sequence_length.png",
                     x_bound=[0, 28],
                     x_tick_interval=4)

    create_histogram(first_db_in_seq, 10, 'Starting Level [dB HL]', 'Number of Sequences',
                     "Starting Level per Sequence",
                     save_path=save_path + "/dataset_sequence_initial_level.png", x_bound=[-40, 120], x_tick_interval=20)

def compute_noise_variance_per_record(record_df, waveform_df, signal_window=(4, 15), noise_start=20.0):
    rows = []
    sig_lo, sig_hi = signal_window

    for _, rec in record_df.iterrows():
        record_id = rec['record_id']
        waveform = get_waveform(record_id, waveform_df)
        if waveform is None or len(waveform) == 0:
            continue

        waveform = np.asarray(waveform, dtype=float)
        epoch = float(rec['epoch'])
        time_axis = np.linspace(0, epoch, len(waveform))

        noise_var = np.nan
        if epoch > noise_start:
            noise_mask = time_axis >= noise_start
            if noise_mask.sum() >= 2:
                noise_var = float(np.nanvar(waveform[noise_mask]))

        signal_var = np.nan
        signal_mask = (time_axis >= sig_lo) & (time_axis <= sig_hi)
        if signal_mask.sum() >= 2:
            signal_var = float(np.nanvar(waveform[signal_mask]))

        if np.isnan(noise_var) and np.isnan(signal_var):
            continue

        ratio = np.nan
        if not np.isnan(noise_var) and not np.isnan(signal_var) and noise_var > 0:
            ratio = signal_var / noise_var

        rows.append({
            'record_id': record_id,
            'db': rec.get('db', np.nan),
            'freq': rec.get('freq', np.nan),
            'epoch': epoch,
            'noise_variance': noise_var,
            'signal_variance': signal_var,
            'snr_ratio': ratio,
        })

    return pd.DataFrame(rows)

def get_freq_graph(namespace, save_path):
    action_df = pd.read_csv(f"../../data/feature_tables/{namespace}/action_table.csv")
    input_data = pd.read_csv(f"../../data/feature_tables/{namespace}/input.csv")
    big_merged_df = pd.merge(input_data, action_df, on="record_id", how='left')
    label_ids = big_merged_df.iloc[:, 14:].idxmax(axis=1)
    create_freq_graph(label_ids, save_path + "/db_adujustments.png")

def create_joint_graphs(sources, save_path):
    create_joint_confusion_matrix(
        model_dict=dict(
                (key, value)
                for path in sources
                for key, value in [
                    (f"{path}_dnn", f"{path}/dnn/pred_spread.csv"),
                    (f"{path}_lstm", f"{path}/lstm/pred_spread.csv"),
                    (f"{path}_transformer", f"{path}/transformer/pred_spread.csv"),
                ]
            ),
        reverse_mapping_path=f"{sources[1]}/lstm/reverse_mapping.json",
        save_path=f"{save_path}/joint_cm.png",
        indices_to_blank=[2, 4, 6, 8, 10]
    )


    create_joint_sequence_accuracy_plot(
        model_dict=dict(
                (key, value)
                for path in sources
                for key, value in [
                    (f"{path}_lstm", f"{path}/lstm/sequence_results.json"),
                    (f"{path}_transformer", f"{path}/transformer/sequence_results.json"),
                ]
            ),
        save_path=f"{save_path}/joint_sequence_accuracy.png"
    )

    create_joint_freq_accuracy_plot(
        model_dict=dict(
                (key, value)
                for path in sources
                for key, value in [
                    (f"{path}_dnn", f"{path}/dnn/freq_results.json"),
                    (f"{path}_lstm", f"{path}/lstm/freq_results.json"),
                    (f"{path}_transformer", f"{path}/transformer/freq_results.json"),
                ]
        ),
        save_path=f"{save_path}/joint_freq_accuracy.png"
    )

def compute_percentile_bound(data, lower=1, upper=99, pad_frac=0.05):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return None

    lo, hi = np.percentile(data, [lower, upper])
    pad = (hi - lo) * pad_frac
    return [float(lo - pad), float(hi + pad)]

def plot_starting_db_level(preprocess_dir, namespace, save_path, session=True):
    with open(f"{preprocess_dir}/toneburst_initial_dbs_{namespace}.json", "r") as f:
        post_click_dbs = json.load(f)

    with open(f"{preprocess_dir}/session_initial_dbs_{namespace}.json", "r") as f:
        click_dbs = json.load(f)

    if post_click_dbs:
        create_histogram(post_click_dbs, 10, 'Starting level [dB HL]', 'Occurrences',
                         f'Post Click Starting DBs ({namespace})',
                         save_path=f"{save_path}/post_click_starting_session_dbs.png", centered=True, x_bound=[-40, 120])
    if click_dbs:
        create_histogram(click_dbs, 10, 'Starting level [dB HL]', 'Occurrences',
                         f'Click Starting DBs ({namespace})', save_path=f"{save_path}/click_starting_session_dbs.png",
                         centered=True, x_bound=[-40, 120])

def plot_signal_noise_variance(record_df, waveform_df, save_path, signal_window=(4, 15), eps=1e-8):
    variance_df = compute_noise_variance_per_record(record_df, waveform_df, signal_window=signal_window)

    log_noise = np.log10(variance_df['noise_variance'].dropna() + eps).tolist()
    log_signal = np.log10(variance_df['signal_variance'].dropna() + eps).tolist()
    snr_ratio = variance_df['snr_ratio'].dropna().tolist()

    create_histogram(log_noise, 0.1,
                     'Noise-Window Variance', 'Number of Records',
                     'Distribution of Noise Variance',
                     x_bound=[-6, 0], x_tick_interval=1, log_x_labels=True,
                     save_path=save_path + "/dataset_noise_variance.png")

    create_histogram(log_signal, 0.1,
                     'Signal-Window Variance', 'Number of Records',
                     'Distribution of Signal Variance',
                     x_bound=[-6, 0], x_tick_interval=1, log_x_labels=True,
                     save_path=save_path + "/dataset_signal_variance.png")

    ratio_bound = compute_percentile_bound(snr_ratio, lower=1, upper=99)
    create_histogram(snr_ratio, 0.5, 'Signal/Noise Variance Ratio', 'Number of Records',
                     'Distribution of SNR Ratio',
                     x_bound=ratio_bound,
                     save_path=save_path + "/dataset_snr_ratio.png")

    return variance_df

def export_comparison_panel(sch_namespace, uw_namespace, preprocess_dir, save_path, session=False):
    _, sch_record_df, _ = get_dataset(sch_namespace)
    _,  uw_record_df,  _ = get_dataset(uw_namespace)

    grouped_dfs_sch = get_grouped_dfs(sch_record_df, ['session_id', 'freq', 'ear'], sort=True,
                                   output=False, save_folder="seattle_childrens",
                                   filter_func=filter_out_ones)

    grouped_dfs_uw = get_grouped_dfs(uw_record_df, ['session_id', 'freq', 'ear'], sort=True,
                                   output=False, save_folder="seattle_childrens",
                                   filter_func=filter_out_ones)


    sch_seq_lengths = list(map(lambda x: len(x), grouped_dfs_sch))
    uw_seq_lengths = list(map(lambda x: len(x), grouped_dfs_uw))

    if session:
        with open(f"{preprocess_dir}/toneburst_initial_dbs_{sch_namespace}.json") as f:
            sch_post_click_dbs = json.load(f)
        with open(f"{preprocess_dir}/toneburst_initial_dbs_{uw_namespace}.json") as f:
            uw_post_click_dbs = json.load(f)
    else:
        sch_post_click_dbs = list(map(lambda x: x['db'].iloc[0], grouped_dfs_sch))
        uw_post_click_dbs =  list(map(lambda x: x['db'].iloc[0], grouped_dfs_uw))

    sch_action_df = pd.read_csv(f"../../data/feature_tables/{sch_namespace}/action_table.csv")
    sch_input_df  = pd.read_csv(f"../../data/feature_tables/{sch_namespace}/input.csv")
    sch_merged    = pd.merge(sch_input_df, sch_action_df, on="record_id", how='left')
    sch_label_ids = sch_merged.iloc[:, 14:].idxmax(axis=1)

    uw_action_df  = pd.read_csv(f"../../data/feature_tables/{uw_namespace}/action_table.csv")
    uw_input_df   = pd.read_csv(f"../../data/feature_tables/{uw_namespace}/input.csv")
    uw_merged     = pd.merge(uw_input_df, uw_action_df, on="record_id", how='left')
    uw_label_ids  = uw_merged.iloc[:, 14:].idxmax(axis=1)

    sch_averages = sch_record_df['averages'].tolist()
    uw_averages = uw_record_df['averages'].tolist()

    create_dataset_comparison_panel(
        sch_seq_lengths=sch_seq_lengths,
        uw_seq_lengths=uw_seq_lengths,
        sch_post_click_dbs=sch_post_click_dbs,
        uw_post_click_dbs=uw_post_click_dbs,
        sch_label_ids=sch_label_ids,
        uw_label_ids=uw_label_ids,
        sch_averages=sch_averages,
        uw_averages=uw_averages,
        save_path=save_path,
        figsize=(43, 18),
        fontsize=40
    )

def export_filter_panel(sch_namespace, uw_namespace, save_path):
    sch_record_df = load_resource(f"db_tables/{sch_namespace}/records_table.csv")
    uw_record_df = load_resource(f"db_tables/{uw_namespace}/records_table.csv")

    create_filter_comparison_panel(
        sch_low_filters=sch_record_df['low_filter'].tolist(),
        uw_low_filters=uw_record_df['low_filter'].tolist(),
        sch_high_filters=sch_record_df['high_filter'].tolist(),
        uw_high_filters=uw_record_df['high_filter'].tolist(),
        save_path=save_path
    )

def load_waveforms_for_records(namespace, record_ids):
    # read the waveform chunks one at a time and keep only the requested records
    record_ids = set(record_ids)
    chunks = []
    for path in sorted(glob.glob(f"../../data/db_tables/{namespace}/waveforms/*.csv")):
        chunk = pd.read_csv(path)
        chunks.append(chunk[chunk['record_id'].isin(record_ids)])
    return pd.concat(chunks, ignore_index=True).set_index('record_id')

def compute_level_traces(namespace, levels, top_n_averages, window, freqs=None, band="sd", n_points=256):
    record_df = load_resource(f"db_tables/{namespace}/records_table.csv")

    top_averages = record_df['averages'].value_counts().head(top_n_averages).index
    selected = record_df[record_df['averages'].isin(top_averages)
                         & record_df['db'].isin(levels)
                         & (record_df['epoch'] + WAVEFORM_START_MS >= window[1])]
    if freqs is not None:
        selected = selected[selected['freq'].isin(freqs)]
    print(f"[{namespace}] top {top_n_averages} averages: {sorted(top_averages.tolist())}, "
          f"{len(selected)} records at {list(levels)} dB")

    waveform_df = load_waveforms_for_records(namespace, selected['record_id'])
    # epochs differ between records, so resample just the analysis window to a fixed number of points
    time_grid = np.linspace(window[0], window[1], n_points)

    traces = []
    for level in levels:
        level_records = selected[selected['db'] == level]
        resampled = []
        for _, rec in level_records.iterrows():
            values = waveform_df.loc[rec['record_id']].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            time_axis = waveform_times(rec['epoch'], len(values))
            resampled.append(np.interp(time_grid, time_axis, values))
        resampled = np.array(resampled)

        mean = resampled.mean(axis=0)
        sd = resampled.std(axis=0, ddof=1)
        half_width = sd if band == "sd" else 1.96 * sd / np.sqrt(len(resampled))
        traces.append({"label": f"{level} dB HL", "n": len(resampled), "x": time_grid,
                       "mean": mean, "lower": mean - half_width, "upper": mean + half_width})
    return traces

def export_waveform_waterfall_panel(sch_namespace, uw_namespace, save_path, levels=(70, 50, 30),
                                    top_n_averages=5, freqs=None, window=(4, 15), band="sd"):
    # band="sd" shows spread between waveforms (mean ± 1 SD); band="ci95" shows uncertainty of the mean
    dataset_traces = {
        "SCH": compute_level_traces(sch_namespace, levels, top_n_averages, window, freqs, band),
        "IHDD": compute_level_traces(uw_namespace, levels, top_n_averages, window, freqs, band),
    }
    create_waveform_waterfall_panel(dataset_traces, save_path=save_path, xlim=window,
                                    band_label="± 1 SD" if band == "sd" else "95% CI of mean")

SEQUENCE_KEY = ['session_id', 'freq', 'ear']

CONTROL_DB = -20

def clean_sequences(record_df):
    # drop the -20 dB no-response controls, then cut each sequence after its last clinician-marked
    # wave V; that waveform is the threshold the sequence should stop on (unmarked sequences drop out)
    record_df = record_df[record_df['db'] != CONTROL_DB].sort_values(['start_time', 'record_id'])
    last_marked = (record_df[record_df['latency_5'].notna()].groupby(SEQUENCE_KEY)['start_time'].max()
                   .rename('last_marked_time'))
    record_df = record_df.join(last_marked, on=SEQUENCE_KEY)
    return record_df[record_df['start_time'] <= record_df['last_marked_time']].drop(columns='last_marked_time')

def get_candidate_sequences(record_df, freq, top_n_buckets=5, bucket_width=100, min_epoch=15, min_length=4,
                            min_levels=3, no_marks_below_threshold=True):
    # sequences where every waveform's averages fall in the dataset's most common buckets
    record_df = record_df.assign(bucket=(record_df['averages'] // bucket_width) * bucket_width)
    top_buckets = record_df['bucket'].value_counts().head(top_n_buckets).index
    record_df = clean_sequences(record_df[record_df['freq'] == freq])

    grouped = record_df.groupby(SEQUENCE_KEY)
    sequences = grouped.agg(n=('db', 'size'), n_levels=('db', 'nunique'), min_epoch=('epoch', 'min'),
                            all_top=('bucket', lambda b: b.isin(top_buckets).all()),
                            threshold=('db', 'last'))
    sequences['lowest_marked'] = record_df[record_df['latency_5'].notna()].groupby(SEQUENCE_KEY)['db'].min()
    keep = ((sequences['n'] >= min_length) & (sequences['n_levels'] >= min_levels)
            & sequences['all_top'] & (sequences['min_epoch'] + WAVEFORM_START_MS >= min_epoch))
    if no_marks_below_threshold:
        # skip sequences where wave V was marked lower than the final (threshold) waveform
        keep &= sequences['lowest_marked'] == sequences['threshold']
    return sequences[keep]

def match_sequences(sch_candidates, uw_candidates, threshold_range, target_length, seed=None):
    # pick the SCH/IHDD pair with the closest threshold, then the closest length, then lengths nearest
    # target_length (so every column is about the same size), then thresholds nearest the category's target
    lo, hi, target = threshold_range
    sch = sch_candidates[sch_candidates['threshold'].between(lo, hi)].reset_index()
    uw = uw_candidates[uw_candidates['threshold'].between(lo, hi)].reset_index()
    pairs = sch.merge(uw, how='cross', suffixes=('_sch', '_uw'))
    pairs['threshold_diff'] = (pairs['threshold_sch'] - pairs['threshold_uw']).abs()
    pairs['length_diff'] = (pairs['n_sch'] - pairs['n_uw']).abs()
    pairs['target_length_diff'] = (pairs['n_sch'] - target_length).abs() + (pairs['n_uw'] - target_length).abs()
    pairs['target_diff'] = (pairs['threshold_sch'] - target).abs() + (pairs['threshold_uw'] - target).abs()
    ranking = ['threshold_diff', 'length_diff', 'target_length_diff', 'target_diff']
    pairs = pairs.sort_values(ranking + ['session_id_uw', 'session_id_sch'])
    if seed is None:
        best = pairs.iloc[0]
    else:
        # draw from every closely matched pair (same threshold gap as the best, lengths within one of
        # each other and of target_length), so reruns can show different sequences
        top = pairs.iloc[0]
        pool = pairs[(pairs['threshold_diff'] == top['threshold_diff']) & (pairs['length_diff'] <= 1)
                     & ((pairs['n_sch'] - target_length).abs() <= 1) & ((pairs['n_uw'] - target_length).abs() <= 1)]
        best = pool.sample(1, random_state=seed).iloc[0] if len(pool) else top
    return ({k: best[f"{k}_sch"] for k in SEQUENCE_KEY + ['n', 'threshold']},
            {k: best[f"{k}_uw"] for k in SEQUENCE_KEY + ['n', 'threshold']})

def build_sequence_panel(namespace, record_df, sequence, time_grid):
    records = clean_sequences(record_df[(record_df['session_id'] == sequence['session_id'])
                                        & (record_df['freq'] == sequence['freq'])
                                        & (record_df['ear'] == sequence['ear'])])
    waveform_df = load_waveforms_for_records(namespace, records['record_id'])

    traces = []
    for i, (_, rec) in enumerate(records.iterrows()):
        values = waveform_df.loc[rec['record_id']].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        time_axis = waveform_times(rec['epoch'], len(values))
        traces.append({"db": rec['db'], "x": time_grid, "y": np.interp(time_grid, time_axis, values),
                       "latency_5": float(rec['latency_5']) if pd.notna(rec['latency_5']) else None,
                       "is_threshold": i == len(records) - 1})

    print(f"[{namespace}] session {sequence['session_id']}, {sequence['ear']} ear, {sequence['freq']} Hz: "
          f"threshold {sequence['threshold']:g} dB, {sequence['n']} waveforms, levels {records['db'].tolist()}")
    return {"traces": traces, "threshold": sequence['threshold']}

def export_sequence_waterfall_panel(sch_namespace, uw_namespace, save_path, freq=2000,
                                    top_n_buckets=5, bucket_width=100, window=(4, 15), n_points=256,
                                    target_length=6, seed=None, pins=None, exclude=None):
    # pins: {("sch" | "uw", column index): (session_id, ear)} keeps a chosen sequence in that panel;
    # exclude: {"sch" | "uw": [(session_id, ear), ...]} keeps sequences out of every panel
    pins, exclude = pins or {}, exclude or {}
    # (lowest threshold, highest threshold, preferred threshold) per column
    categories = {
        "Severe Loss (75–90 dB HL)": (75, 90, 80),
        "Moderate Loss (45–70 dB HL)": (45, 70, 55),
        "Normal Hearing (≤ 20 dB HL)": (-np.inf, 20, 10),
    }
    time_grid = np.linspace(window[0], window[1], n_points)
    sch_record_df = load_resource(f"db_tables/{sch_namespace}/records_table.csv")
    uw_record_df = load_resource(f"db_tables/{uw_namespace}/records_table.csv")
    sch_candidates = get_candidate_sequences(sch_record_df, freq, top_n_buckets, bucket_width, window[1])
    uw_candidates = get_candidate_sequences(uw_record_df, freq, top_n_buckets, bucket_width, window[1])

    def restrict(candidates, dataset, column):
        keys = list(zip(candidates.index.get_level_values('session_id'), candidates.index.get_level_values('ear')))
        pin = pins.get((dataset, column))
        keep = [key == pin if pin else key not in exclude.get(dataset, []) for key in keys]
        return candidates[keep]

    sch_row, uw_row = [], []
    for column, threshold_range in enumerate(categories.values()):
        sch_seq, uw_seq = match_sequences(restrict(sch_candidates, "sch", column),
                                          restrict(uw_candidates, "uw", column),
                                          threshold_range, target_length, seed)
        sch_row.append(build_sequence_panel(sch_namespace, sch_record_df, sch_seq, time_grid))
        uw_row.append(build_sequence_panel(uw_namespace, uw_record_df, uw_seq, time_grid))

    # column order follows `categories` (left to right); session details are printed above
    create_sequence_waterfall_panel([sch_row, uw_row], row_labels=["SCH", "IHDD"],
                                    save_path=save_path, xlim=window)

def export_dataset_plots(namespace):
    session_df, record_df, waveform_df = get_dataset(namespace)
    save_path = f"../../data/feature_tables/{namespace}/plots"
    plot_averages(record_df, save_path)
    plot_session_lengths(record_df, save_path)
    plot_session_lengths_by_freq(record_df, save_path)
    plot_bone_ear_distrib(record_df, save_path)
    plot_db_freq_distribution(record_df, save_path)
    plot_sequence_length(record_df, save_path)
    plot_starting_db_level("../../exported_features/stimulus_change/preprocess", namespace, save_path)
    get_freq_graph(namespace, save_path)
    plot_signal_noise_variance(record_df, waveform_df, save_path)

create_joint_graphs(["../../exported_features/stimulus_change/results/train", "../../exported_features/stimulus_change/cross_set"], "../../data/feature_tables/sch/plots")

export_comparison_panel(
    sch_namespace="sch",
    uw_namespace="uw",
    preprocess_dir="../../exported_features/stimulus_change/preprocess",
    save_path="../../data/feature_tables/sch/plots/dataset_comparison_panel.png"
)

export_filter_panel(
    sch_namespace="sch",
    uw_namespace="uw",
    save_path="../../data/feature_tables/sch/plots/filter_comparison_panel.png"
)

export_waveform_waterfall_panel(
    sch_namespace="sch",
    uw_namespace="uw",
    save_path="../../data/feature_tables/sch/plots/waveform_waterfall_panel.png"
)

# sequences hand-picked from the matched candidates: (dataset, column) -> (session_id, ear)
export_sequence_waterfall_panel(
    sch_namespace="sch",
    uw_namespace="uw",
    save_path="../../data/feature_tables/sch/plots/sequence_waterfall_panel.png",
    pins={("sch", 0): (2947, "Left"), ("uw", 0): (165, "Right"),
          ("sch", 1): (2446, "Right"), ("uw", 1): (83, "Right"),
          ("sch", 2): (2536, "Right"), ("uw", 2): (400, "Left")}
)

export_dataset_plots("sch")
export_dataset_plots("uw")
#create_binary_confusion_matrix("../../exported_features/wave_5/results/train/dnn/pred_spread.csv", "../../exported_features/wave_5/results/train/dnn/dnn_2_cm.png")