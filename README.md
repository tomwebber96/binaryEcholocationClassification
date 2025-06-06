# Binary Echolocation Classification

# UPDATING . . .

![image](logo.png)

*Created with DALL·E: "Deep learning classifier for dolphin echolocation".*

## Summary
This deep learning tool aims to aid in odontocete species identification by classifying recorded echolocation clicks. The example provided here is a binary classification for Risso's dolphins in UK waters, but the model can be re-trained for other species and regions as required. The Risso's dolphin models have been devloped for data (re)sampled at 96 kHz or 250 kHz in UK waters.  Trained on towed multi-channel visually verified single species data, with the 96 kHz model tested on novel static data. See PAPER LINK for details. The models provided in this repository can be used either within exisitng PAMGuard workflows, or run on PAMGuard click detections outside of PAMGuard using python. A .py script is provided which could be modified to enable classification of normalised waveform snipits created by other impulsive noise detectors. While available, note the 250 kHz model is untested on novel data and within PAMGuard.


## Running within PAMGuard
First clone this binaryClickClassifier repo. Within /IGNORE/PAMGuard/ is an example .psfx for running the 96kHz classifier on single channel data. The actual input .wav example sampled at 96 kHz can be downloaded here. The click detector in this case is run on raw input data. Where collected at >96 kHz, re-sample data within PAMGuard and run click detector on resampled data. Most importantly, include the two 10 kHz high pass butteworth filters. The trigger threshold here is set to the default 10 dB but can be updated as required.
 
Within the deep learning module, direct PAMGuard to the 96kHz_PB_model.zip. PAMGuard will automatically select the correct settings for the model. A 250kHz_PB_model.zip is availble but is mostly untested.
 
Ouputs will be stored in PAMGuards deep learning .pgdf's. It is recommended to use a mean classification score over a given time bin with thresholds tested for each usecase.
 
## Running from a windows .bat file
Ensure you have a python installation (3.9.12 - tested and working - https://www.python.org/downloads/release/python-3912/). This can be within an anaconda distribution. 
 
Ensure you have run a relevent PAMGuard click detector through your data (currently this works for single channel data. Multiple channels can work but will require some editing to the runClass.py). See IGNORE/PAMGuard/clickDetector.psfx for an example working with single channel data sampled at 500 kHz, resampled to 96 kHz for click detection, with a 10 kHz pre and trigger high pass filter.
 
Generate .mat file copies of .pgdf's. See IGNORE/pgdf_to_mat.m
 
Clone this binaryClickClassifier repo and copy in the processed PAMGuard data such that it exists in a subfolder e.g. Site_data with the subfolder "binary" containing daily folders of .mat files. 
 ```
 └── binaryClickClassifier /
    ├── Site_name/
    │   ├── binary/
    │   │   └── 20240106/
    │   │       ├── Click_Detector_Click_Detector_Clicks_20240106_000000.mat
    │   │       ├── Click_Detector_Click_Detector_Clicks_20240106_010000.mat
```

Next, run createVirEnv.bat. It will ask to be directed to your python installation. If unsure run *where python* in cmd or your anaconda installation. Pip installation may require admin rights. 
 
runClassifer.bat will then ask for which model you want to select, 96kHz or 250kHz. Select option 1 or 2. Any errors should become obvious and most likely due to the python installation on the system.
 
The .bat file will create a new folder called classified with a copy of the .mat files with the suffix_classified, containing a normalised copy of the waveforms and individual prediction scores. It will also generate a .csv file with a mean classification score in 5-minute time bins.
 
If the classifer is stopped, simply re-start, and it will pick up where it left off, by checking through previous outputs. 
 
## Re-training
Full models in .h5 formats are avaialble in */models* for 96 and 250 kHz data. Either add in more data at leisure e.g. */IGNORE/aditionalTraining.py*. Or take the model arcitecture, e.g. run summary(model) and train from scratch. E.g. for a new species or new region.   
