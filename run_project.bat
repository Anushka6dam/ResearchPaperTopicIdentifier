
@echo off
title Research Paper Topic Identifier
cd /d "%~dp0"

echo ==========================================
echo   Research Paper Topic Identification
echo ==========================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Please check that the venv folder exists.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: app.py was not found.
    pause
    exit /b 1
)

echo Starting the website...
echo.
echo Keep this window open while using the website.
echo Open http://127.0.0.1:5000 in your browser.
echo.

start "" http://127.0.0.1:5000
"venv\Scripts\python.exe" app.py

echo.
echo The website has stopped.
pause