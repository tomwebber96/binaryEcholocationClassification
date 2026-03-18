# Binary Echolocation Classification

# UPDATING . . .

![image](logo.png)

*Created with DALL·E: "Deep learning classifier for dolphin echolocation".*

## Summary
This deep learning tool aims to aid in odontocete species identification by classifying recorded echolocation clicks. The example provided here is a binary classification for Risso's dolphins in UK waters, but the model can be re-trained for other species and regions as required. The Risso's dolphin models have been developed for data (re)sampled at 96 kHz or 250 kHz in UK waters.  Trained on towed multi-channel visually verified single species data, with the 96 kHz model tested on novel static data. See PAPER LINK for details. The models provided in this repository can be used either within existing PAMGuard workflows, or run on PAMGuard click detections outside of PAMGuard using python. A .py script is provided which could be modified to enable classification of normalised waveform snippits created by other impulsive noise detectors. While available, note the 250 kHz model is untested on new data (SOON TO CHANGE).

## Running within PAMGuard
First clone this binaryClickClassifier repository. Within /IGNORE/PAMGuard/ is an example .psfx for running the 96kHz classifier on single channel data. The click detector in this case is run on raw input data. Where collected at >96 kHz, re-sample data within PAMGuard and run click detector on resampled data. Most importantly, include the two 10 kHz high pass butterworth filters. The trigger threshold here is set to the default 10 dB but can be updated as required.
 
Within the deep learning module, direct PAMGuard to the 96kHz_PB_model.zip. PAMGuard will automatically select the correct settings for the model. A 250kHz_PB_model.zip is available but is mostly untested.
 
Outputs will be stored in PAMGuard's deep learning .pgdf's. It is recommended to use a mean classification score over a given time bin with thresholds tested for each use case. See PAMGuard help (https://www.pamguard.org/olhelp/utilities/BinaryStore/docs/matlabandr.html) for opening binary files for further data analysis.

Differences in exact prediction scores can occasioanlly occur between running in PAMGuard and the .bat file. These are typically <0.001, but in the cases of waveforms which are shorter than 64 samples (96 kHz model) or 128 (250 kHz model), the centering and padding methods differ slightly. Mean difference of 0.001 tested over 1000 click waveforms.
 
## Running from a windows .bat file
The .h5 models used within the .bat file approach were last trained using python 3.10.10. The specific LSTMs within these models were last supported with Tensorflow 2.12.0. This version of tensorflow is not supported after Python 3.12. For native GPU support in windows, Tensorflow must be 2.10 or older, as such, python 3.10 is the most recent which will support this tensorflow version. GPU compatiability has been tested on an NVIDA CUDA enabled GPU only, requiring CUDA 11.7 (https://developer.nvidia.com/cuda-11-7-0-download-archive) and cuDNN libraries 8.9 (https://developer.nvidia.com/rdp/cudnn-archive). This may differ depending on your GPU. See https://developer.nvidia.com/cudnn#section-how-cudnn-works for help.

Ensure you have a python installation (3.10.0 - tested and working - https://www.python.org/downloads/release/python-3120/). This can be within an anaconda distribution. 
 
Ensure you have run a relevent PAMGuard click detector through your data (currently this works for single channel data. Multiple channels can work but will require some editing to the runClassPGDF.py for selecting one or all channels, and merging the predictions across channels). See IGNORE/PAMGuard/clickDetector.psfx for an example working with single channel data sampled at 500 kHz, resampled to 96 kHz for click detection, with a 10 kHz pre and trigger high pass filter.

Clone this binaryClickClassifier repo and copy in the processed PAMGuard data such that it exists in a subfolder e.g. Site_data with the subfolder "binary" containing daily folders of .mat/ .pgdf files. Many different sites can be run simultaneously, simply compy the below structure for each site within the parent directory.
 ```
 └── binaryClickClassifier /
    ├── Site_name/
    │   ├── binary/
    │   │   └── 20240106/
    │   │       ├── Click_Detector_Click_Detector_Clicks_20240106_000000.pgdf
    │   │       ├── Click_Detector_Click_Detector_Clicks_20240106_010000.pgdf
```

Next, run createVirEnv.bat. It will ask to be directed to your python installation. If unsure run *where python* in cmd or your anaconda installation. Python's Pip installation may require admin rights. These versions have been tested and work with the current setup, but if a different GPU or other versions of packages are installed, further installs/ updates maybe required for your specific needs. 

Use the batch script *runClassifier_PGDF.bat* to work from .pgdf's directly. Currently individual predictions can be stored as .npy files with one .npy file per .pgdf file. .npy files can also be opened within R (https://cran.r-project.org/web/packages/RcppCNPy/vignettes/UsingReticulate.pdf)depending on the users workflow preferences. Updates may take place which will aim to produce .npy files that also store normalized and clipped waveforms along with original PAMGuard waveforms.
 

Run the runClassifier_PGDF.bat. It will ask for which model you want to select, 96kHz or 250kHz. Select option 1 or 2. Any errors should become obvious and most likely due to the python installation on the system. The .bat will also ask if you want to save the individual predictions to .npy files, and what PAMGuard click detection threshold (signal excess) you with to use. The current PAMGuard .psfx has a default 10dB threshold. The .bat file allows 10,12,14, and 16 dB thresholds to be implimented by predicting and classifying based on clicks which meet these thresholds. These can be edited within the .py file if required.
 
The .bat file will create a new folder called classified with a copy of the .npy files, containing the individual prediction scores added to the data already within the PAMGuard .pgdf files. It will also generate a .csv file with a mean classification score in 5-minute time bins, along with total number of clicks, number of positive, and number of negative (based on <0.5 being negative).
 
If the classifier is stopped, simply re-start, and it will pick up where it left off (in whole days), by checking through previous outputs in the .csv. 


## Re-training
Full models in .h5 formats are available in */models* for 96 and 250 kHz data. Either add in more data at leisure , or take the model architecture, e.g. run summary(model) and train from scratch. E.g. for a new species or new region. Scripts for further training or training from scratch may be devloped at a later date.
