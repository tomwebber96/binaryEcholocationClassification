@echo off

REM Prompt user for the Python installation path
set /p PYTHON_PATH="Enter the full path to your Python executable (e.g., C:\Python39\python.exe): "

REM Check if the provided Python executable exists
if not exist "%PYTHON_PATH%" (
    echo The specified Python executable was not found. Please check the path and try again.
    pause
    exit /b
)

REM Get the directory of the current .bat file
set "SCRIPT_DIR=%~dp0"

REM Navigate to the directory of the .bat file
cd /d "%SCRIPT_DIR%"

REM Inform the user that the environment creation is starting
echo Creating the virtual environment 'Gg_Env'...

REM Create the Python virtual environment named Gg_Env
"%PYTHON_PATH%" -m venv Gg_Env
if errorlevel 1 (
    echo Failed to create virtual environment. Please check your Python installation.
    pause
    exit /b
)

REM Check if virtual environment was created successfully
if not exist "Gg_Env\Scripts\activate" (
    echo Failed to create virtual environment. Ensure the specified Python installation supports venv.
    pause
    exit /b
)

REM Inform the user that the virtual environment has been created
echo Virtual environment 'Gg_Env' has been created successfully.

REM Activate the virtual environment
echo Activating the virtual environment...
call Gg_Env\Scripts\activate
if errorlevel 1 (
    echo Failed to activate the virtual environment.
    pause
    exit /b
)

REM Inform the user that package installation will begin
echo Installing packages from requirements.txt...

REM Check if requirements.txt exists and install packages
if exist "requirements.txt" (
    echo Installing packages...
    pip install -r requirements.txt >> install_log.txt 2>&1
    if errorlevel 1 (
        echo Failed to install packages from requirements.txt. If you encounter "Access is denied" errors, try running the batch file as Administrator.
        echo Please right-click the batch file and select "Run as Administrator" to install the packages.
        pause
        exit /b
    )
    echo Installed packages from requirements.txt.
) else (
    echo No requirements.txt found. Skipping package installation.
)

REM Verify installation of all packages listed in requirements.txt
echo Verifying installation of packages...

REM Loop through each line in requirements.txt and check if it is installed
for /f "delims=" %%i in (requirements.txt) do (
    REM Check if the package is installed
    pip show %%i >nul 2>&1
    if errorlevel 1 (
        echo Package %%i is missing or failed to install.
    ) else (
        echo Package %%i is installed successfully.
    )
)

REM Deactivate the virtual environment
echo Deactivating the virtual environment...
deactivate
if errorlevel 1 (
    echo Failed to deactivate the virtual environment.
    pause
    exit /b
)

REM Final message to notify user that the process is complete
echo The virtual environment has been successfully set up.
echo Press any key to exit...

REM Keep the window open and wait for user input to close
pause
