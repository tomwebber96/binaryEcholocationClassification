@echo off
REM Get directory path
SET base_file_location=%~dp0
REM Strip trailing backslash to prevent quote-escaping issues
SET base_file_location=%base_file_location:~0,-1%
ECHO Drive Path: %base_file_location%

REM Ask user to choose a model (1 for 96kHz, 2 for 250kHz)
ECHO Select model to use:
ECHO 1. 96kHz
ECHO 2. 250kHz
SET /P model_choice=Choose [1 or 2]: 
IF "%model_choice%"=="1" SET model_choice=96kHz
IF "%model_choice%"=="2" SET model_choice=250kHz
ECHO You chose: %model_choice%

REM Ask user for signal excess threshold
ECHO.
ECHO Select minimum signal excess threshold (dB):
ECHO 1. 10 dB
ECHO 2. 12 dB
ECHO 3. 14 dB
ECHO 4. 16 dB
ECHO 5. All (calculates stats for all thresholds)
SET /P se_choice=Choose [1, 2, 3, 4 or 5]: 
IF "%se_choice%"=="1" SET signal_excess=10
IF "%se_choice%"=="2" SET signal_excess=12
IF "%se_choice%"=="3" SET signal_excess=14
IF "%se_choice%"=="4" SET signal_excess=16
IF "%se_choice%"=="5" SET signal_excess=all
ECHO Signal excess threshold: %signal_excess%

REM Ask user whether to save prediction .npy files
ECHO.
ECHO Save prediction scores to .npy files?
ECHO 1. Yes
ECHO 2. No
SET /P write_predictions=Choose [1 or 2]: 
IF "%write_predictions%"=="1" SET write_predictions=yes
IF "%write_predictions%"=="2" SET write_predictions=no
ECHO Write predictions: %write_predictions%

REM Activate environment
CALL "%base_file_location%\Gg_Env\Scripts\activate.bat"

REM Change directory to where your script is located
cd "%base_file_location%"

REM Check if activation was successful
echo Activating virtual environment...
echo Environment activated.

REM Set CUDA and cuDNN paths (adjust paths if needed)
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\libnvvp;%PATH%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Verify Python executable and version
echo Python executable: %CONDA_EXE%
python -c "import sys; print('Python executable:', sys.executable); print('Python version:', sys.version)"

REM Run the Python script
python "%base_file_location%\runClassPGDF.py" "%model_choice%" "%base_file_location%" "%signal_excess%" "%write_predictions%"
pause