@echo off
echo Audio Dataset Processor
echo ---------------------
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python and try again.
    exit /b 1
)

REM Check for pydub installation
python -c "import pydub" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: pydub module not found. Installing...
    pip install pydub
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install pydub. Please install it manually:
        echo pip install pydub
        exit /b 1
    )
)

REM Default directories
set INPUT_DIR=dataset\raw
set WAV_DIR=dataset\wav
set CHUNKS_DIR=dataset\chunks
set CHUNK_LENGTH=10000
set OVERLAP=0

echo This tool will convert all audio files in a directory to WAV format
echo and split them into chunks for processing.
echo.
echo Default settings:
echo - Input directory: %INPUT_DIR%
echo - WAV output directory: %WAV_DIR%
echo - Chunks output directory: %CHUNKS_DIR%
echo - Chunk length: %CHUNK_LENGTH% ms (10 seconds)
echo - Overlap: %OVERLAP% ms (0 seconds)
echo.

set /p CUSTOM_SETTINGS="Use custom settings? (y/n): "
if /i "%CUSTOM_SETTINGS%"=="y" (
    set /p INPUT_DIR="Input directory: "
    set /p WAV_DIR="WAV output directory: "
    set /p CHUNKS_DIR="Chunks output directory: "
    set /p CHUNK_LENGTH="Chunk length (ms): "
    set /p OVERLAP="Overlap (ms): "
)

echo.
echo Processing with the following settings:
echo - Input directory: %INPUT_DIR%
echo - WAV output directory: %WAV_DIR%
echo - Chunks output directory: %CHUNKS_DIR%
echo - Chunk length: %CHUNK_LENGTH% ms
echo - Overlap: %OVERLAP% ms
echo.

set /p CONFIRM="Continue? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Operation cancelled.
    exit /b
)

echo.
echo Starting processing...
echo.

python audio_processor.py --input "%INPUT_DIR%" --wav_output "%WAV_DIR%" --chunks_output "%CHUNKS_DIR%" --chunk_length %CHUNK_LENGTH% --overlap %OVERLAP%

echo.
echo Processing complete.
echo.
pause 