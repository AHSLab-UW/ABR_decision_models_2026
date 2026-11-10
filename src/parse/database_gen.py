import json
import os
import re
import pandas as pd
import numpy as np
from src.common.plot_util import create_histogram
from src.common.data_util import *

class DatasetParser:
    def __init__(self, namespace):
        self.namespace = namespace
        self.input_dir = f"../../data/database/{namespace}"
        self.output_dir = f"../../data/db_tables/{namespace}"
        self.plot_dir=f"../../data/feature_tables/{namespace}/plots"
        self.exported_data_dir=f"../../exported_features/stimulus_change/preprocess"

        self.bone_conduction = 0
        self.total_records = 0
        self.click_record = 0
        self.click_dbs = []
        self.post_click_dbs = []

        self.sessions_data = []
        self.records_data = []
        self.waveforms_data = []

        self.session_id_counter = 1
        self.record_id_counter = 1

    def get_file_paths(self):
        data = []
        for dirpath, _, filenames in os.walk(self.input_dir):
            for fname in filenames:
                if fname.endswith(".Txt"):
                    data.append(os.path.join(dirpath, fname))
        return data

    def parse_session(self, file_path):
        print(f"[{self.namespace}] Parsing: {file_path}")

        file_indicator = file_path.split("\\")[-1].split(".")[0]
        split_file_name_data = file_indicator.split("_")

        if len(split_file_name_data) < 3:
            return

        patient_str = split_file_name_data[1]
        session_num = split_file_name_data[2]

        if len(split_file_name_data) > 3:
            if any(x in split_file_name_data for x in ['A', 'C', 'L']):
                return

        current_session_id = self.session_id_counter
        self.sessions_data.append({
            'session_id': current_session_id,
            'patient_id': patient_str,
            'patient_session_num': session_num,
            'file': file_indicator
        })
        self.session_id_counter += 1

        with open(file_path, "r") as f:
            line_iterator = iter(f.readlines())

            try:
                next_line = next(line_iterator)
            except StopIteration:
                return

            click_reached = False
            first_record_seen = False
            first_toneburst_seen = False

            while next_line:
                if not next_line.__contains__("AEP"):
                    next_line = next(line_iterator, None)
                    continue

                stripped_line = next_line.strip().split("\t")
                test_type = get_value(stripped_line[0])
                date = parse_date(get_value(stripped_line[1]))
                channels_num = get_value(stripped_line[2])
                epoch = get_value(stripped_line[3])
                points = get_value(stripped_line[4])
                pre_post_time = get_value(stripped_line[5])
                blocked_time = get_value(stripped_line[6])
                comments = get_value(stripped_line[7])

                next(line_iterator, None)
                next_line = next(line_iterator, None)

                if next_line is None: break

                stripped_line = next_line.strip().split("\t")
                ear = get_value(stripped_line[0])
                transducer = get_value(stripped_line[1])
                frequency = get_middle_value(stripped_line[2], 2)
                self.total_records += 1

                if 'Bone Oscillator' in transducer:
                    self.bone_conduction += 1
                    next_line = next(line_iterator, None)
                    continue

                dB = get_middle_value(get_value(stripped_line[4]), 0)

                if not first_record_seen:
                    self.click_dbs.append(dB)
                    first_record_seen = True

                if frequency is None:
                    if not click_reached:
                        click_reached = True
                        self.click_record += 1
                    next_line = next(line_iterator, None)
                    continue

                if not first_toneburst_seen:
                    self.post_click_dbs.append(dB)
                    first_toneburst_seen = True

                click_reached = False

                polarity = get_value(stripped_line[3])
                rate = get_value(stripped_line[5])
                masking = get_value(stripped_line[6])

                banks = []
                next(line_iterator, None)
                next_line = next(line_iterator, "").strip()

                while len(next_line) > 0:
                    stripped_line = next_line.split("\t")
                    bank_dict = {
                        'channel_num': get_value(stripped_line[0]),
                        'bank_num': get_value(stripped_line[1]),
                        'averages': get_value(stripped_line[2]),
                        'rejected': get_value(stripped_line[3]),
                        'gain': get_value(stripped_line[4]),
                        'low_filter': get_value(stripped_line[5]),
                        'high_filter': get_value(stripped_line[6]),
                        'notch_filter': get_value(stripped_line[7]),
                        'bank_comment': get_value(stripped_line[8]) if len(stripped_line) > 8 else ""
                    }
                    banks.append(bank_dict)
                    next_line = next(line_iterator, "").strip()

                next(line_iterator, None)
                next_line = next(line_iterator, "").strip()
                for i in range(len(banks)):
                    stripped_line = next_line.split("\t")
                    banks[i]['latency_5'] = get_value(stripped_line[5]) if len(stripped_line) > 5 else None
                    next_line = next(line_iterator, "").strip()

                next(line_iterator, None)
                next_line = next(line_iterator, "").strip()
                for i in range(len(banks)):
                    stripped_line = next_line.split("\t")
                    banks[i]['amplitude_5'] = get_value(stripped_line[5]) if len(stripped_line) > 5 else None
                    next_line = next(line_iterator, "").strip()

                next(line_iterator, None)
                next_line = next(line_iterator, "").strip()
                waveform_values = []
                while len(next_line) > 0:
                    waveform_values.append(list(map(float, re.split(r'\s+', next_line))))
                    next_line = next(line_iterator, "").strip()

                if not waveform_values:
                    next_line = next(line_iterator, None)
                    continue

                waveform_values = np.array(waveform_values)

                bank_index = 0
                if len(banks) == 2:
                    bank_index = 1 if ear == "Right" else 0
                elif len(banks) == 4:
                    bank_index = 2
                elif len(banks) == 8:
                    bank_index = 2 if ear == "Left" else 6

                target_bank = banks[bank_index]
                waveform_specific_values = waveform_values[:, bank_index]

                if max(waveform_specific_values) - min(waveform_specific_values) > 5:
                    next_line = next(line_iterator, None)
                    continue
                if int(frequency) <= 250:
                    next_line = next(line_iterator, None)
                    continue
                if int(target_bank['averages']) < 750:
                    next_line = next(line_iterator, None)
                    continue

                record_row = {
                    'record_id': self.record_id_counter,
                    'session_id': current_session_id,
                    'test_type': test_type,
                    'start_time': date,
                    'channels': channels_num,
                    'epoch': epoch,
                    'points': points,
                    'pre_post_time': pre_post_time,
                    'blocked_time': blocked_time,
                    'comments': comments,
                    'ear': ear,
                    'transducer': transducer,
                    'freq': frequency,
                    'polarity': polarity,
                    'db': dB,
                    'rate': rate,
                    'masking': masking,
                    'channel_num': target_bank['channel_num'],
                    'bank_num': target_bank['bank_num'],
                    'averages': target_bank['averages'],
                    'rejected': target_bank['rejected'],
                    'gain': target_bank['gain'],
                    'low_filter': target_bank['low_filter'],
                    'high_filter': target_bank['high_filter'],
                    'notch_filter': target_bank['notch_filter'],
                    'bank_comment': target_bank['bank_comment'],
                    'latency_5': target_bank['latency_5'],
                    'amplitude_5': target_bank['amplitude_5']
                }
                self.records_data.append(record_row)

                waveform_row = {'record_id': self.record_id_counter}
                for idx, value in enumerate(waveform_specific_values):
                    waveform_row[f'value_{idx}'] = value
                self.waveforms_data.append(waveform_row)

                self.record_id_counter += 1
                next_line = next(line_iterator, None)

    def parse_dataset(self):
        print(f"Parsing dataset {self.namespace}")

        for file_path in self.get_file_paths():
            self.parse_session(file_path)

        print(f"Converting to DataFrames...")
        sessions_df = pd.DataFrame(self.sessions_data)
        records_df = pd.DataFrame(self.records_data)
        waveforms_df = pd.DataFrame(self.waveforms_data)

        os.makedirs(self.output_dir, exist_ok=True)
        waveforms_dir = f"{self.output_dir}/waveforms"
        os.makedirs(waveforms_dir, exist_ok=True)
        clean_files(waveforms_dir)

        print(f"Exporting CSVs...")
        sessions_df.to_csv(f"{self.output_dir}/sessions_table.csv", index=False)
        records_df.to_csv(f"{self.output_dir}/records_table.csv", index=False)
        split_df_across_files(waveforms_df, 25000, f"{waveforms_dir}/waveforms_table")
        #waveforms_df.to_csv(f"{self.output_dir}/waveforms_table.csv", index=False)

        print(
            f"[{self.namespace}] Successfully exported {len(sessions_df)} sessions, {len(records_df)} records, and {len(waveforms_df)} waveforms.")
        print(f"[{self.namespace}] Bone Conduction: {self.bone_conduction} / {self.total_records}")
        print(f"[{self.namespace}] Click: {self.click_record} / {self.total_records}")

        click_dbs_clean = list(map(int, self.click_dbs))
        post_click_dbs_clean = list(map(int, self.post_click_dbs))

        print(f"[{self.namespace}] Click dB range:", min(click_dbs_clean) if click_dbs_clean else "N/A",
              max(click_dbs_clean) if click_dbs_clean else "N/A")
        print(f"[{self.namespace}] Post-click dB range:", min(post_click_dbs_clean) if post_click_dbs_clean else "N/A",
              max(post_click_dbs_clean) if post_click_dbs_clean else "N/A")

        with open(f"{self.exported_data_dir}/toneburst_initial_dbs_{self.namespace}.json", "w") as f:
            json.dump(post_click_dbs_clean, f)

        with open(f"{self.exported_data_dir}/session_initial_dbs_{self.namespace}.json", "w") as f:
            json.dump(click_dbs_clean, f)

        if post_click_dbs_clean:
            create_histogram(post_click_dbs_clean, 10, 'Starting level [dB HL]', 'Occurrences',
                             f'Post Click Starting DBs ({self.namespace})', save_path=f"{self.plot_dir}/post_click_starting_dbs.png", centered=True, x_bound=[-40, 120])
        if click_dbs_clean:
            create_histogram(click_dbs_clean, 10, 'Starting level [dB HL]', 'Occurrences',
                             f'Click Starting DBs ({self.namespace})', save_path=f"{self.plot_dir}/click_starting_dbs.png", centered=True, x_bound=[-40, 120])

sch_parser = DatasetParser("sch")
sch_parser.parse_dataset()
uw_parser = DatasetParser("uw")
uw_parser.parse_dataset()