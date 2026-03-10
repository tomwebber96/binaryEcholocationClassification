
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:47:54 2024

@author: Thomas Webber
"""
# Import dependencies.
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



# Waveform normalisation
def normalize_waveform(waveform):
    """Normalize a waveform with mean 0 and std 1."""
    mean = np.mean(waveform)
    std = np.std(waveform)
    if std == 0:
        return waveform  # Avoid division by zero if std is zero
    return (waveform - mean) / std

# Find last entry in working CSV to avoid double effort in case script needs to be restarted
def get_last_entry_date(csv_file_path):
    """Read the CSV file and return the latest date found."""
    if not os.path.isfile(csv_file_path):
        return None

    last_date = None
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        for row in reader:
            try:
                # Use %z to handle timezone in the datetime string
                datetime_obj = dt.strptime(row[0], '%Y-%m-%d %H:%M:%S%z').date()
                if last_date is None or datetime_obj > last_date:
                    last_date = datetime_obj
            except ValueError as e:
                print(f"Error parsing date for row: {row[0]} - {e}")
    return last_date

# Convert MATLAB datenum to Python datetime
def matlab_datenum_to_datetime(datenum):
    """Convert MATLAB datenum into a Python datetime object."""
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)

def convert_datetime_to_timestamp(click_detector_obj):
    if hasattr(click_detector_obj, 'date') and isinstance(click_detector_obj.date, datetime):
        # Convert datetime to Unix timestamp (seconds)
        click_detector_obj.date = int(click_detector_obj.date.timestamp())  # Convert to timestamp
    return click_detector_obj


# Round time down to the nearest 5-minute interval - i.e. 5 min time chunk is start of time chunk- 00:04:59 to 00:00:00
def round_to_nearest_5_minutes(dt):
    discard = timedelta(
        minutes=dt.minute % 5,
        seconds=dt.second,
        microseconds=dt.microsecond
    )
    rounded = dt - discard
    return rounded.replace(tzinfo=dt.tzinfo)  # preserve timezone




def pad_waveforms(data, max_length):
    padded_waveforms = []
    removed_indices = []  # List to store indices of removed waveforms

    for idx, waveform in enumerate(data):
        try:
            # Convert waveform to a numeric array and flatten
            waveform = np.asarray(waveform, dtype=np.float32).flatten()
        except (ValueError, TypeError):
            print(f"Non-numeric waveform at index {idx}. Removing...")
            removed_indices.append(idx)
            continue

        # Check for NaN values and remove the waveform if found
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


"""
PAMGuard Binary Writer
======================
Creates a new .pgdf file identical to the input, but with a float32
'prediction' value appended to each data chunk.

Usage:
    from pamguard_writer import write_pgdf_with_predictions

    write_pgdf_with_predictions(
        input_path='path/to/input.pgdf',
        output_path='path/to/output.pgdf',
        predictions=predictions_array  # 1D numpy array or list, one per detection
    )
"""

CHUNK_INFO_SIZE = 8  # INT32 length + INT32 identifier
ENDIAN = '>'         # PAMGuard default: big-endian


def _read_chunk_info(fp):
    """Read the 8-byte chunk header. Returns (length, identifier) or (None, None) at EOF."""
    raw = fp.read(CHUNK_INFO_SIZE)
    if len(raw) < CHUNK_INFO_SIZE:
        return None, None
    length, identifier = struct.unpack(f'{ENDIAN}ii', raw)
    return length, identifier


def _pack_chunk_info(length, identifier):
    """Pack a chunk header to bytes."""
    return struct.pack(f'{ENDIAN}ii', length, identifier)

def write_pgdf_with_predictions(input_path, output_path, pgdf_data, predictions):
    predictions = list(predictions)
    ENDIAN = '>'

    DATA_FLAG_FIELDS = ['TIMEMILLISECONDS','TIMENANOSECONDS','CHANNELMAP','UID','STARTSAMPLE',
                        'SAMPLEDURATION','FREQUENCYLIMITS','MILLISDURATION','TIMEDELAYSECONDS',
                        'HASBINARYANNOTATIONS','HASSEQUENCEMAP','HASNOISE','HASSIGNAL','HASSIGNALEXCESS']

    with open(input_path, 'rb') as fin:
        file_bytes = fin.read()

    output = bytearray(file_bytes)

    for i, (obj, pred) in enumerate(zip(pgdf_data, predictions)):
        # Calculate offset of click data start within chunk body
        pos = 8   # skip millis INT64
        flags_raw = struct.unpack_from(f'{ENDIAN}H', output, obj._start_pos + 8)[0]
        pos += 2  # flag_bitmap INT16
        set_flags = [DATA_FLAG_FIELDS[j] for j in range(len(DATA_FLAG_FIELDS)) if flags_raw & (1 << j)]

        if 'TIMENANOSECONDS' in set_flags: pos += 8
        if 'CHANNELMAP' in set_flags: pos += 4
        if 'UID' in set_flags: pos += 8
        if 'STARTSAMPLE' in set_flags: pos += 8
        if 'SAMPLEDURATION' in set_flags: pos += 4
        if 'FREQUENCYLIMITS' in set_flags: pos += 8
        if 'MILLISDURATION' in set_flags: pos += 4
        if 'TIMEDELAYSECONDS' in set_flags:
            n = struct.unpack_from(f'{ENDIAN}h', output, obj._start_pos + pos)[0]
            pos += 2 + n * 4
        if 'HASSEQUENCEMAP' in set_flags: pos += 4
        if 'HASNOISE' in set_flags: pos += 4
        if 'HASSIGNAL' in set_flags: pos += 4
        if 'HASSIGNALEXCESS' in set_flags: pos += 4
        pos += 4  # data_length INT32

        # trigger_map INT32 (4 bytes), then type INT16
        type_offset = obj._start_pos + pos + 4
        new_type = np.int16(1) if pred >= 0.5 else np.int16(0)
        struct.pack_into(f'{ENDIAN}h', output, type_offset, int(new_type))

    with open(output_path, 'wb') as fout:
        fout.write(output)

    print(f"  Written {len(predictions)} type classifications to: {output_path}")
    
def main(base_file_location, model_choice, write_predictions=False):
    print(f"The base folder location is: {base_file_location}")
    print(f"Selected model {model_choice}")

    # Check if base_file_location exists
    if not os.path.exists(base_file_location):
        print(f"Error: Base file location does not exist: {base_file_location}")
        return

    # Set WD
    os.chdir(base_file_location)

    # List available devices
    devices = tf.config.list_physical_devices()
    print("Available devices:", devices)

    # Load model
    if model_choice == "96kHz":
        modelLocation = os.path.join(base_file_location, "models", "96_binaryPadded.h5")
    elif model_choice == "250kHz":
        modelLocation = os.path.join(base_file_location, "models", "250_binaryPadded.h5")
    else:
        print(f"Invalid model choice: {model_choice}")
        return

    if model_choice == "96kHz":
        max_length = 64
    elif model_choice == "250kHz":
        max_length = 128

    print(f"Attempting to load model from: {modelLocation}")

    model = None

    if not os.path.exists(modelLocation):
        print(f"Error: The model file cannot be found: {modelLocation}")
        return

    try:
        model = load_model(modelLocation, compile=False, custom_objects={})
        print(f"Model loaded successfully from: {modelLocation}")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Load sites (folders, and remove non-site folders)
    exclude_dirs = ['models', '$RECYCLE.BIN', 'System Volume Information', 'Cuda', 'Gg_env', 'overlappingWAVs', 'IGNORE']
    sites = [f for f in os.listdir(base_file_location)
             if os.path.isdir(os.path.join(base_file_location, f))
             and f.lower() not in (dir_name.lower() for dir_name in exclude_dirs)]

    log_file_path = os.path.join(base_file_location, 'processing_log.txt')
    print("Logging:", log_file_path)
    with open(log_file_path, 'a') as log_file:

        # Main loop
        for site in sites:
            site_path = os.path.join(base_file_location, site)
            log_file.write(f"Starting Site: {site} at {dt.now()}\n")
            print("Loading Site:", site)

            total_waveforms = 0
            pgdf_file_count = 0

            # Clear last entry date variable
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
                    writer.writerow(['datetime_obj', 'meanClass', 'nClicks', 'nPositive', 'nNegative'])

            # Construct the path to the 'binary' folder within the site
            binary_folder_path = os.path.join(site_path, 'binary')

            classified_folder_path = os.path.join(site_path, 'classified')
            if not os.path.exists(classified_folder_path):
                os.makedirs(classified_folder_path)

            # Check if the 'binary' folder exists
            if os.path.exists(binary_folder_path) and os.path.isdir(binary_folder_path):
                # Iterate through all daily folders in the 'binary' directory
                for daily_folder in os.listdir(binary_folder_path):
                    daily_folder_path = os.path.join(binary_folder_path, daily_folder)

                    # Check if it is a directory
                    if os.path.isdir(daily_folder_path):
                        folder_date_str = daily_folder[:8]
                        folder_date = dt.strptime(folder_date_str, '%Y%m%d')

                        if 'last_entry_date' in locals():
                            if last_entry_date and folder_date.date() < last_entry_date:
                                print(f"Skipping folder {daily_folder_path} as it is before the last recorded date.")
                                continue

                        print("Classifying:", daily_folder_path)
                        start_time = time.time()

                        # Initialize storage for the 5-minute bins
                        bin_data = {}

                        # List all .pgdf files in the daily folder
                        for file in os.listdir(daily_folder_path):
                            if file.endswith('.pgdf') and 'Click_Detector_Click_Detector_Clicks' in file:
                                pgdf_file_count += 1
                                pgdf_file_path = os.path.abspath(os.path.join(daily_folder_path, file))
                                try:
                                    # Load the .pgdf file
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

                                    print(f"{file}:")

                                    waveform_list = []
                                    for obj in pgdf_data:
                                        waveform = obj.wave.flatten()
                                        waveform = waveform.reshape(-1, 1)
                                        waveform_time = obj.date
                                        waveform_list.append((waveform, waveform_time))

                                    waveform_array = np.array([item[0] for item in waveform_list], dtype=object)
                                    normalized_waveforms = [normalize_waveform(waveform) for waveform in waveform_array]
                                    waveforms_norm = pad_waveforms(np.array(normalized_waveforms, dtype=object), max_length)[0]

                                    # Predict using the model
                                    predictions = model.predict(waveforms_norm).flatten().astype(np.float32)
                                    total_waveforms += len(predictions)

                                    n_positive = int(np.sum(predictions >= 0.5))
                                    n_negative = len(predictions) - n_positive
                                    print(f"  {len(predictions)} waveforms → {n_positive} positive, {n_negative} negative ({100*n_positive/len(predictions):.1f}%)")

                                    # Write new pgdf with predictions into classified/yyyymmdd/
                                    if write_predictions:
                                        classified_daily_folder = os.path.abspath(os.path.join(classified_folder_path, daily_folder))
                                        if not os.path.exists(classified_daily_folder):
                                            os.makedirs(classified_daily_folder)
                                        output_pgdf_path = os.path.abspath(os.path.join(classified_daily_folder, file))
                                        write_pgdf_with_predictions(pgdf_file_path, output_pgdf_path, pgdf_data, predictions)

                                    # Process each waveform, group into 5-minute intervals
                                    for (waveform, waveform_time), prediction in zip(waveform_list, predictions):
                                        bin_time = round_to_nearest_5_minutes(waveform_time)

                                        if bin_time not in bin_data:
                                            bin_data[bin_time] = {
                                                'nClicks': 0,
                                                'nPositive': 0,
                                                'nNegative': 0,
                                                'predictions': []
                                            }

                                        bin_data[bin_time]['nClicks'] += 1
                                        bin_data[bin_time]['nPositive'] += 1 if prediction >= 0.5 else 0
                                        bin_data[bin_time]['nNegative'] += 1 if prediction < 0.5 else 0
                                        bin_data[bin_time]['predictions'].append(prediction)

                                except Exception as e:
                                    traceback.print_exc()
                                    print(f"Error processing file {pgdf_file_path}: {e}")

                        # After processing all files in daily folder, write 5-minute bins to CSV
                        with open(csv_file_path, mode='a', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            for bin_time, data in sorted(bin_data.items()):
                                meanClass = np.mean(data['predictions'])
                                writer.writerow([bin_time, meanClass, data['nClicks'], data['nPositive'], data['nNegative']])

                        print(f"  Written {len(bin_data)} 5-minute bins to CSV")
                        iteration_time = time.time() - start_time
                        print(f"  Classification for daily folder took {iteration_time:.4f} seconds")

                # After processing all daily folders, save the complete CSV file
                shutil.copy(csv_file_path, complete_csv_file_path)
                print(f"CSV file saved as: {complete_csv_file_path}")

            # Log the site summary
            log_file.write(f"Finished Site: {site} at {dt.now()}\n")
            log_file.write(f"Total .pgdf files processed: {pgdf_file_count}\n")
            log_file.write(f"Total waveforms processed: {total_waveforms}\n")
            log_file.write("--------------------------------------------------\n")
            log_file.flush()

# The entry point of the script

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Missing required arguments.")
        print("Usage: python runClass.py <model_choice> <base_file_location> <write_predictions>")
        sys.exit(1)

    model_choice = sys.argv[1].strip('"')
    base_file_location = sys.argv[2].strip('"')
    write_predictions = sys.argv[3].strip('"').lower() == 'yes' if len(sys.argv) > 3 else False

    print(base_file_location)
    print(model_choice)
    print(f"Write predictions to pgdf: {write_predictions}")

    main(base_file_location, model_choice, write_predictions)


