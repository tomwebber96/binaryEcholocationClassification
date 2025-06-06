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
import tensorflow as tf
import numpy as np
import csv
import shutil
from keras.models import load_model
from scipy.io import savemat, loadmat
from datetime import datetime, timedelta

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
            datetime_obj = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
            if last_date is None or datetime_obj > last_date:
                last_date = datetime_obj
    return last_date

# Convert MATLAB datenum to Python datetime
def matlab_datenum_to_datetime(datenum):
    """Convert MATLAB datenum into a Python datetime object."""
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)

# Round time to the nearest 5-minute interval
def round_to_nearest_5_minutes(dt):
    discard = timedelta(minutes=dt.minute % 5, seconds=dt.second, microseconds=dt.microsecond)
    dt -= discard
    if discard >= timedelta(minutes=2.5):
        dt += timedelta(minutes=5)
    return dt



def pad_waveforms(data, max_length):
    padded_waveforms = []
    removed_indices = []  # List to store indices of removed waveforms

    for idx, waveform in enumerate(data):
        waveform = waveform.flatten()  # Flatten to ensure we have a 1D array

        # Check for NaN values and remove the waveform if found
        if np.any(np.isnan(waveform)):
            print(f"Found NaN in waveform at index {idx}. Removing...")
            removed_indices.append(idx)  # Store index of removed waveform
            continue

        current_length = len(waveform)
        if current_length > max_length:
            # Find the index of the maximum positive value
            max_index = np.argmax(waveform)
            # Calculate the start and end indices to center the max value
            start_index = max(0, max_index - max_length // 2)
            end_index = start_index + max_length
            
            # Adjust indices if they are out of bounds
            if end_index > current_length:
                end_index = current_length
                start_index = end_index - max_length
            
            # Truncate the waveform to the calculated range
            padded_waveform = waveform[start_index:end_index]
        
        elif current_length < max_length:
            # Find the index of the maximum positive value
            max_index = np.argmax(waveform)
            # Calculate padding needed to center the max value
            pad_before = max(0, max_length // 2 - max_index)
            pad_after = max_length - (current_length + pad_before)
            
            # If padding is not balanced, adjust to ensure target length
            if pad_after < 0:
                pad_before += pad_after
                pad_after = 0
            
            # Pad the waveform to the calculated range
            padded_waveform = np.pad(waveform, (pad_before, pad_after), 'constant')
        
        else:
            # If waveform is already of the target length, no need to modify
            padded_waveform = waveform
        
        padded_waveforms.append(padded_waveform)

    return np.array(padded_waveforms, dtype=np.float32).reshape(-1, max_length, 1), removed_indices

def main(base_file_location, model_choice):
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
    	return

    
    # Debugging: Print the model path to check if it's correct
    print(f"Attempting to load model from: {modelLocation}")
    
    # Check if the model path exists
    if not os.path.exists(modelLocation):
        print(f"Error: The model file cannot be found: {modelLocation}")
        return

    # Load the model if the path is valid
    try:
        model = load_model(modelLocation)
        print(f"Model loaded successfully from: {modelLocation}")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return
    
    model = load_model(modelLocation)

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
            log_file.write(f"Starting Site: {site} at {datetime.now()}\n")
            print("Loading Site:", site)
            
            total_waveforms = 0
            mat_file_count = 0
    
            # Clear last entry date variable
            if 'last_entry_date' in locals():
                del last_entry_date
    
            csv_file_path = os.path.join(site_path, f'{site}_recordingClass.csv')
            complete_csv_file_path = os.path.join(site_path, f'{site}_recordingClass_complete.csv')
    
            if os.path.isfile(complete_csv_file_path):
                print(f"Skipping site {site} as {complete_csv_file_path} already exists.")
                continue  # Skip to the next site if the complete file is present
    
            if os.path.isfile(csv_file_path):
                last_entry_date = get_last_entry_date(csv_file_path)
                print(f"Starting from {last_entry_date}.")
            else:
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
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
                        folder_date = datetime.strptime(folder_date_str, '%Y%m%d')
                        
                        if 'last_entry_date' in locals():
                            if last_entry_date and folder_date.date() < last_entry_date:
                                print(f"Skipping folder {daily_folder_path} as it is before the last recorded date.")
                                continue

                        print("Classifying:", daily_folder_path)
                        start_time = time.time()

                        # Initialize storage for the 5-minute bins
                        bin_data = {}
                        
                        # List all .mat files in the daily folder
                        for file in os.listdir(daily_folder_path):
                            if file.endswith('.mat'):
                                mat_file_count += 1
                                mat_file_path = os.path.join(daily_folder_path, file)
                                try:
                                    # Load the .mat file
                                    mat_data = loadmat(mat_file_path)
                                    clkstruct = mat_data['binarydata']
    
                                    # Extract the start time from the first waveform
                                    start_time_num = clkstruct[0, 0]['date'][0, 0]  # Example: array([[738285.38541667]])
                                    process_start_time = matlab_datenum_to_datetime(start_time_num)
                                    print(f"Start time for {file}: {process_start_time}")

                                    waveform_list = []

                                    # Extract waveforms
                                    for i in range(clkstruct.shape[1]):
                                        waveform = clkstruct[0, i]['wave']
                                        waveform_time_num = clkstruct[0, i]['date'][0, 0]
                                        waveform_time = matlab_datenum_to_datetime(waveform_time_num)
                                        waveform_list.append((waveform, waveform_time))
                                                                            
                                    waveform_array = np.array([item[0] for item in waveform_list], dtype=object)
                                    normalized_waveforms = [normalize_waveform(waveform) for waveform in waveform_array]
                                    waveforms_norm = pad_waveforms(np.array(normalized_waveforms, dtype=object), max_length)[0]

                                    # Predict using the model
                                    predictions = model.predict(waveforms_norm).flatten().astype(np.float32)
                                    total_waveforms += len(predictions)
                                  
                                    # Add predictions to the clkstruct
                                    original_dtype = clkstruct.dtype
                                    new_dtype = np.dtype([(name, original_dtype[name]) for name in original_dtype.names] +
                                                         [('pred', object), ('wave_norm', object)] )
                                    new_clkstruct = np.empty(clkstruct.shape, dtype=new_dtype)
                                    for name in original_dtype.names:
                                        new_clkstruct[name] = clkstruct[name]
                                    new_clkstruct['pred'] = predictions
                                    new_clkstruct['wave_norm'] = normalized_waveforms

                                    # Save the classified file
                                    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
                                    new_file_path = os.path.join(classified_folder_path, f"{base_name}_classified.mat")
                                    savemat(new_file_path, {'binarydata': new_clkstruct})

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

                                        # Aggregate data in the current bin
                                        bin_data[bin_time]['nClicks'] += 1
                                        bin_data[bin_time]['nPositive'] += 1 if prediction >= 0.5 else 0
                                        bin_data[bin_time]['nNegative'] += 1 if prediction < 0.5 else 0
                                        bin_data[bin_time]['predictions'].append(prediction)

                                except Exception as e:
                                    num_waveforms = clkstruct.shape[1] if 'clkstruct' in locals() else 'unknown'
                                    print(f"Error processing file {mat_file_path}: {e}")
                                    print(f"Number of waveforms in the structure: {num_waveforms}")

                        # After processing the waveforms, write data for each 5-minute bin to the CSV
                        with open(csv_file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            for bin_time, data in sorted(bin_data.items()):
                                meanClass = np.mean(data['predictions'])
                                writer.writerow([bin_time, meanClass, data['nClicks'], data['nPositive'], data['nNegative']])

                        iteration_time = time.time() - start_time
                        print(f"Classification took {iteration_time:.4f} seconds")
        
                # After processing all daily folders, save the complete CSV file
                shutil.copy(csv_file_path, complete_csv_file_path)
                print(f"CSV file saved as: {complete_csv_file_path}")

            # Log the site summary
            log_file.write(f"Finished Site: {site} at {datetime.now()}\n")
            log_file.write(f"Total .mat files processed: {mat_file_count}\n")
            log_file.write(f"Total waveforms processed: {total_waveforms}\n")
            log_file.write("--------------------------------------------------\n")
            log_file.flush()  # Ensure log entry is written to file

# The entry point of the script

if __name__ == "__main__":
    base_file_location = sys.argv[2].strip('"')
    #base_file_location = f'"{base_file_location}"'
    print(base_file_location)
    model_choice = sys.argv[1].strip('"')
    #model_choice = f'"{model_choice}"'
    print(model_choice)
        
    if len(sys.argv) < 3:
    	print("Error: Missing required arguments.")
    	print("Usage: python runClass.py <base_file_location> <model_choice>")
    	sys.exit(1)

    main(base_file_location, model_choice)

