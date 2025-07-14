@echo off
echo ================================================
echo DBD Auto Skill Check - Launcher
echo ================================================

REM Check if we're in the correct directory
if not exist "app.py" (
    echo Error: app.py not found!
    echo Please make sure you're running this from the correct directory.
    pause
    exit /b 1
)

REM Check if python-embed directory exists (for embedded version)
if exist "python-embed" (
    echo Using embedded Python...
    "python-embed\python.exe" app.py
) else (
    echo Using system Python...
    python app.py
    if errorlevel 1 (
        echo.
        echo Error: Failed to start with 'python' command.
        echo Trying 'python3'...
        python3 app.py
        if errorlevel 1 (
            echo.
            echo Error: Could not start the application.
            echo Please make sure Python is installed and accessible.
            echo.
            echo You can also try:
            echo 1. Installing Python from https://python.org
            echo 2. Running: python -m pip install -r requirements.txt
            echo 3. Then running: python app.py
        )
    )
)

echo.
echo Application has stopped.
pause
