# -*- coding: utf-8 -*-
"""
Created on Aug 29 09:47:54 2024
@author: Thomas Webber - SAMS
"""
import sys
import os
import scipy.io
import time
import struct
import tensorflow as tf
import numpy as np
import csv
import shutil
import h5py, json
import logging
import warnings
import traceback

import datetime
if not hasattr(datetime, 'UTC'):
    datetime.UTC = datetime.timezone.utc

from keras.models import load_model
from tensorflow.keras.models import model_from_json
from scipy.io import savemat, loadmat
from datetime import datetime as dt, timedelta
import pypamguard

sys.stderr = open(os.devnull, 'w')
logging.getLogger('pypamguard').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*ChunkLengthMismatch.*")


def normalize_waveform(waveform):
    mean = np.mean(waveform)
    std = np.std(waveform)
    if std == 0:
        return waveform
    return (waveform - mean) / std


def get_last_entry_date(csv_file_path):
    if not os.path.isfile(csv_file_path):
        return None
    last_date = None
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        for row in reader:
            try:
                datetime_obj = dt.strptime(row[0], '%Y-%m-%d %H:%M:%S%z').date()
                if last_date is None or datetime_obj > last_date:
                    last_date = datetime_obj
            except ValueError as e:
                print(f"Error parsing date for row: {row[0]} - {e}")
    return last_date


def round_to_nearest_5_minutes(dt):
    discard = timedelta(
        minutes=dt.minute % 5,
        seconds=dt.second,
        microseconds=dt.microsecond
    )
    rounded = dt - discard
    return rounded.replace(tzinfo=None)  # strip timezone for clean CSV output


def pad_waveforms(data, max_length):
    padded_waveforms = []
    removed_indices = []

    for idx, waveform in enumerate(data):
        try:
            waveform = np.asarray(waveform, dtype=np.float32).flatten()
        except (ValueError, TypeError):
            print(f"Non-numeric waveform at index {idx}. Removing...")
            removed_indices.append(idx)
            continue

        if np.any(np.isnan(waveform)):
            print(f"Found NaN in waveform at index {idx}. Removing...")
            removed_indices.append(idx)
            continue

        current_length = len(waveform)
        if current_length > max_length:
            max_index = np.argmax(waveform)
            start_index = max(0, max_index - max_length // 2)
            end_index = start_index + max_length
            if end_index > current_length:
                end_index = current_length
                start_index = end_index - max_length
            padded_waveform = waveform[start_index:end_index]
        elif current_length < max_length:
            max_index = np.argmax(waveform)
            pad_before = max(0, max_length // 2 - max_index)
            pad_after = max_length - (current_length + pad_before)
            if pad_after < 0:
                pad_before += pad_after
                pad_after = 0
            padded_waveform = np.pad(waveform, (pad_before, pad_after), 'constant')
        else:
            padded_waveform = waveform

        padded_waveforms.append(padded_waveform)

    return np.array(padded_waveforms, dtype=np.float32).reshape(-1, max_length, 1), removed_indices


def save_predictions_npy(output_path, pgdf_data, predictions):
    """Save [uid, millis, prediction] array to .npy file."""
    uids   = np.array([obj.uid    for obj in pgdf_data], dtype=np.int64)
    millis = np.array([obj.millis for obj in pgdf_data], dtype=np.int64)
    preds  = np.array(predictions, dtype=np.float32)
    result = np.column_stack([uids, millis, preds])
    np.save(output_path, result)
    print(f"  Saved predictions to: {output_path}  (shape {result.shape})")


SIGNAL_EXCESS_THRESHOLDS = [10, 12, 14, 16]


def main(base_file_location, model_choice, signal_excess_choice, write_predictions=False):
    print(f"The base folder location is: {base_file_location}")
    print(f"Selected model: {model_choice}")
    print(f"Signal excess threshold: {signal_excess_choice} dB")

    if not os.path.exists(base_file_location):
        print(f"Error: Base file location does not exist: {base_file_location}")
        return

    os.chdir(base_file_location)

    devices = tf.config.list_physical_devices()
    print("Available devices:", devices)

    if model_choice == "96kHz":
        modelLocation = os.path.join(base_file_location, "models", "96_binaryPadded.h5")
        max_length = 64
    elif model_choice == "250kHz":
        modelLocation = os.path.join(base_file_location, "models", "250_binaryPadded.h5")
        max_length = 128
    else:
        print(f"Invalid model choice: {model_choice}")
        return

    print(f"Attempting to load model from: {modelLocation}")

    if not os.path.exists(modelLocation):
        print(f"Error: The model file cannot be found: {modelLocation}")
        return

    try:
        model = load_model(modelLocation, compile=False, custom_objects={})
        print(f"Model loaded successfully from: {modelLocation}")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    all_mode = (signal_excess_choice.lower() == "all")
    if not all_mode:
        single_threshold = int(signal_excess_choice)

    exclude_dirs = ['models', '$RECYCLE.BIN', 'System Volume Information', 'Cuda', 'Gg_env', 'overlappingWAVs', 'IGNORE']
    sites = [f for f in os.listdir(base_file_location)
             if os.path.isdir(os.path.join(base_file_location, f))
             and f.lower() not in (d.lower() for d in exclude_dirs)]

    log_file_path = os.path.join(base_file_location, 'processing_log.txt')
    print("Logging:", log_file_path)

    with open(log_file_path, 'a') as log_file:

        for site in sites:
            site_path = os.path.join(base_file_location, site)
            log_file.write(f"Starting Site: {site} at {dt.now()}\n")
            print("Loading Site:", site)

            total_waveforms = 0
            pgdf_file_count = 0

            if 'last_entry_date' in locals():
                del last_entry_date

            csv_file_path = os.path.join(site_path, f'{site}_recordingClass.csv')
            complete_csv_file_path = os.path.join(site_path, f'{site}_recordingClass_complete.csv')

            if os.path.isfile(complete_csv_file_path):
                print(f"Skipping site {site} as {complete_csv_file_path} already exists.")
                continue

            if os.path.isfile(csv_file_path):
                last_entry_date = get_last_entry_date(csv_file_path)
                print(f"Starting from {last_entry_date}.")
            else:
                with open(csv_file_path, mode='w', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    if all_mode:
                        header = ['datetime_obj']
                        for t in SIGNAL_EXCESS_THRESHOLDS:
                            header += [
                                f'meanClass_{t}dB',
                                f'mean_top10pct_{t}dB',
                                f'mean_top5pct_{t}dB',
                                f'nClicks_{t}dB',
                                f'nPositive_{t}dB',
                                f'nNegative_{t}dB',
                            ]
                        writer.writerow(header)
                    else:
                        writer.writerow(['datetime_obj', 'meanClass', 'mean_top10pct', 'mean_top5pct',
                                         'nClicks', 'nPositive', 'nNegative'])

            binary_folder_path = os.path.join(site_path, 'binary')
            classified_folder_path = os.path.join(site_path, 'classified')
            if not os.path.exists(classified_folder_path):
                os.makedirs(classified_folder_path)

            if os.path.exists(binary_folder_path) and os.path.isdir(binary_folder_path):

                for daily_folder in os.listdir(binary_folder_path):
                    daily_folder_path = os.path.join(binary_folder_path, daily_folder)

                    if os.path.isdir(daily_folder_path):
                        folder_date_str = daily_folder[:8]
                        folder_date = dt.strptime(folder_date_str, '%Y%m%d')

                        if 'last_entry_date' in locals():
                            if last_entry_date and folder_date.date() < last_entry_date:
                                print(f"Skipping folder {daily_folder_path} as it is before the last recorded date.")
                                continue

                        print("Classifying:", daily_folder_path)
                        start_time = time.time()

                        bin_data = {}

                        click_files = [f for f in os.listdir(daily_folder_path)
                                       if f.endswith('.pgdf') and 'Click_Detector_Click_Detector_Clicks' in f]

                        for file in click_files:
                            pgdf_file_count += 1
                            pgdf_file_path = os.path.abspath(os.path.join(daily_folder_path, file))

                            try:
                                with open(os.devnull, 'w') as devnull:
                                    old_stdout = sys.stdout
                                    sys.stdout = devnull
                                    try:
                                        pgdf_file = pypamguard.load_pamguard_binary_file(pgdf_file_path)
                                    finally:
                                        sys.stdout = old_stdout

                                if pgdf_file is None or pgdf_file.data is None:
                                    print(f"Warning: Could not load or empty data in {pgdf_file_path}, skipping.")
                                    continue

                                pgdf_data = np.array(pgdf_file.data)

                                # Extract signal_excess for every click
                                signal_excess_vals = np.array(
                                    [obj.signal_excess if obj.signal_excess is not None else np.nan
                                     for obj in pgdf_data],
                                    dtype=np.float32
                                )

                                print(f"{file}: {len(pgdf_data)} clicks loaded")

                                # -----------------------------------------------------------
                                # Helper: predict on a boolean-masked subset, return full-
                                # length array (NaN for excluded clicks)
                                # -----------------------------------------------------------
                                def predict_subset(mask):
                                    subset_objs = pgdf_data[mask]
                                    if len(subset_objs) == 0:
                                        return np.full(len(pgdf_data), np.nan, dtype=np.float32)

                                    waveform_list = []
                                    for obj in subset_objs:
                                        waveform = obj.wave.flatten().reshape(-1, 1)
                                        waveform_list.append((waveform, obj.date))

                                    waveform_array = np.array([item[0] for item in waveform_list], dtype=object)
                                    normalized = [normalize_waveform(w) for w in waveform_array]
                                    waveforms_norm = pad_waveforms(np.array(normalized, dtype=object), max_length)[0]
                                    preds_subset = model.predict(waveforms_norm).flatten().astype(np.float32)

                                    full_preds = np.full(len(pgdf_data), np.nan, dtype=np.float32)
                                    full_preds[mask] = preds_subset
                                    return full_preds

                                # -----------------------------------------------------------
                                # Single threshold mode
                                # -----------------------------------------------------------
                                if not all_mode:
                                    mask = signal_excess_vals >= single_threshold
                                    n_kept = int(mask.sum())
                                    print(f"  Signal excess >= {single_threshold} dB: {n_kept}/{len(pgdf_data)} clicks kept")

                                    full_preds = predict_subset(mask)
                                    subset_preds = full_preds[mask]
                                    total_waveforms += n_kept

                                    if n_kept > 0:
                                        n_positive = int(np.sum(subset_preds >= 0.5))
                                        n_negative = n_kept - n_positive
                                        print(f"  {n_kept} waveforms → {n_positive} positive, {n_negative} negative "
                                              f"({100*n_positive/n_kept:.1f}%)")

                                    if write_predictions:
                                        classified_daily_folder = os.path.abspath(
                                            os.path.join(classified_folder_path, daily_folder))
                                        if not os.path.exists(classified_daily_folder):
                                            os.makedirs(classified_daily_folder)
                                        npy_name = os.path.splitext(file)[0] + '_predictions.npy'
                                        npy_path = os.path.join(classified_daily_folder, npy_name)
                                        save_predictions_npy(npy_path, pgdf_data[mask], subset_preds)

                                    for obj, prediction in zip(pgdf_data[mask], subset_preds):
                                        bin_time = round_to_nearest_5_minutes(obj.date)
                                        if bin_time not in bin_data:
                                            bin_data[bin_time] = {
                                                'nClicks': 0, 'nPositive': 0,
                                                'nNegative': 0, 'predictions': []
                                            }
                                        bin_data[bin_time]['nClicks'] += 1
                                        bin_data[bin_time]['nPositive'] += 1 if prediction >= 0.5 else 0
                                        bin_data[bin_time]['nNegative'] += 1 if prediction < 0.5 else 0
                                        bin_data[bin_time]['predictions'].append(prediction)

                                # -----------------------------------------------------------
                                # All-thresholds mode — predict ONCE on lowest threshold (10 dB)
                                # then filter predictions per threshold for CSV/npy output
                                # -----------------------------------------------------------
                                else:
                                    min_threshold = min(SIGNAL_EXCESS_THRESHOLDS)
                                    base_mask = signal_excess_vals >= min_threshold
                                    n_kept = int(base_mask.sum())
                                    print(f"  All mode: predicting on signal excess >= {min_threshold} dB "
                                          f"({n_kept}/{len(pgdf_data)} clicks)")

                                    full_preds = predict_subset(base_mask)
                                    total_waveforms += n_kept

                                    # Log counts per threshold using the single set of predictions
                                    for t in SIGNAL_EXCESS_THRESHOLDS:
                                        t_mask = signal_excess_vals >= t
                                        t_preds = full_preds[t_mask]
                                        n_t = int(t_mask.sum())
                                        if n_t > 0:
                                            n_pos = int(np.sum(t_preds >= 0.5))
                                            mean_cls = float(np.mean(t_preds))
                                            print(f"    >= {t} dB: {n_t} clicks | {n_pos} positive | mean class {mean_cls:.3f}")

                                    # Save single .npy with uid, millis, signal_excess, prediction
                                    # for all clicks >= 10 dB — thresholds can be re-applied later
                                    if write_predictions:
                                        classified_daily_folder = os.path.abspath(
                                            os.path.join(classified_folder_path, daily_folder))
                                        if not os.path.exists(classified_daily_folder):
                                            os.makedirs(classified_daily_folder)
                                        npy_name = os.path.splitext(file)[0] + '_predictions_all.npy'
                                        npy_path = os.path.join(classified_daily_folder, npy_name)
                                        base_objs = pgdf_data[base_mask]
                                        uids   = np.array([o.uid    for o in base_objs], dtype=np.int64)
                                        millis = np.array([o.millis for o in base_objs], dtype=np.int64)
                                        se     = signal_excess_vals[base_mask]
                                        preds  = full_preds[base_mask]
                                        np.save(npy_path, np.column_stack([uids, millis, se, preds]))
                                        print(f"  Saved predictions to: {npy_path}")

                                    # Group into 5-minute bins, filtering per threshold
                                    for obj in pgdf_data:
                                        bin_time = round_to_nearest_5_minutes(obj.date)
                                        if bin_time not in bin_data:
                                            bin_data[bin_time] = {t: [] for t in SIGNAL_EXCESS_THRESHOLDS}

                                    for t in SIGNAL_EXCESS_THRESHOLDS:
                                        t_mask = signal_excess_vals >= t
                                        for obj, prediction in zip(pgdf_data[t_mask], full_preds[t_mask]):
                                            bin_time = round_to_nearest_5_minutes(obj.date)
                                            bin_data[bin_time][t].append(float(prediction))

                            except Exception as e:
                                traceback.print_exc()
                                print(f"Error processing file {pgdf_file_path}: {e}")

                        # -----------------------------------------------------------
                        # Write 5-minute bins to CSV
                        # -----------------------------------------------------------
                        with open(csv_file_path, mode='a', newline='') as csv_file:
                            writer = csv.writer(csv_file)

                            if not all_mode:
                                for bin_time, data in sorted(bin_data.items()):
                                    preds = np.array(data['predictions'])
                                    meanClass = float(np.mean(preds))
                                    p90 = np.percentile(preds, 90)
                                    p95 = np.percentile(preds, 95)
                                    mean_top10pct = float(np.mean(preds[preds >= p90])) if np.any(preds >= p90) else float(p90)
                                    mean_top5pct  = float(np.mean(preds[preds >= p95])) if np.any(preds >= p95) else float(p95)
                                    writer.writerow([bin_time, meanClass, mean_top10pct, mean_top5pct,
                                                     data['nClicks'], data['nPositive'], data['nNegative']])
                            else:
                                for bin_time, tdata in sorted(bin_data.items()):
                                    row = [bin_time]
                                    for t in SIGNAL_EXCESS_THRESHOLDS:
                                        preds = np.array(tdata[t], dtype=np.float32)
                                        if len(preds) == 0:
                                            row += [0.0, 0.0, 0.0, 0, 0, 0]
                                        else:
                                            meanClass = float(np.mean(preds))
                                            p90 = np.percentile(preds, 90)
                                            p95 = np.percentile(preds, 95)
                                            mean_top10 = float(np.mean(preds[preds >= p90])) if np.any(preds >= p90) else float(p90)
                                            mean_top5  = float(np.mean(preds[preds >= p95])) if np.any(preds >= p95) else float(p95)
                                            n_pos = int(np.sum(preds >= 0.5))
                                            row += [meanClass, mean_top10, mean_top5,
                                                    len(preds), n_pos, len(preds) - n_pos]
                                    writer.writerow(row)

                        print(f"  Written {len(bin_data)} 5-minute bins to CSV")
                        iteration_time = time.time() - start_time
                        print(f"  Classification for daily folder took {iteration_time:.4f} seconds")

                shutil.copy(csv_file_path, complete_csv_file_path)
                print(f"CSV file saved as: {complete_csv_file_path}")

            log_file.write(f"Finished Site: {site} at {dt.now()}\n")
            log_file.write(f"Total .pgdf files processed: {pgdf_file_count}\n")
            log_file.write(f"Total waveforms processed: {total_waveforms}\n")
            log_file.write("--------------------------------------------------\n")
            log_file.flush()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Error: Missing required arguments.")
        print("Usage: python runClassPGDF.py <model_choice> <base_file_location> <signal_excess_choice> [write_predictions]")
        sys.exit(1)

    model_choice           = sys.argv[1].strip('"')
    base_file_location     = sys.argv[2].strip('"')
    signal_excess_choice   = sys.argv[3].strip('"').lower()
    write_predictions      = sys.argv[4].strip('"').lower() == 'yes' if len(sys.argv) > 4 else False

    print(f"Write predictions to .npy: {write_predictions}")

    main(base_file_location, model_choice, signal_excess_choice, write_predictions)
