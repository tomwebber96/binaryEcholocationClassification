@echo off

REM Get directory path
SET base_file_location=%~dp0
ECHO Dive Path: %base_file_location%

REM Ask user to choose a model (1 for 96kHz, 2 for 250kHz)
ECHO Select model to use:
ECHO 1. 96kHz
ECHO 2. 250kHz
SET /P model_choice=Choose [1 or 2]:

IF "%model_choice%"=="1" SET model_choice=96kHz
IF "%model_choice%"=="2" SET model_choice=250kHz

ECHO You chose: %model_choice%

REM Activate environment
CALL "%base_file_location%Gg_Env\Scripts\activate.bat"

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

REM Run the Python script with the base file location and model choice as arguments
python "%base_file_location%runClass.py" "%base_file_location%" "%model_choice%"

pause
