@echo off
echo Installing ComfyUI OpenCV Overlays dependencies...
echo.

REM Navigate to the custom node directory
cd /d "%~dp0"

REM Find ComfyUI's Python executable
set "COMFYUI_PYTHON=..\..\..\python_embeded\python.exe"
if not exist "%COMFYUI_PYTHON%" (
    set "COMFYUI_PYTHON=..\..\..\python\python.exe"
)
if not exist "%COMFYUI_PYTHON%" (
    set "COMFYUI_PYTHON=python"
    echo Using system Python...
) else (
    echo Using ComfyUI's Python: %COMFYUI_PYTHON%
)

echo.
echo Installing requirements...
"%COMFYUI_PYTHON%" -m pip install --upgrade pip
"%COMFYUI_PYTHON%" -m pip install -r requirements.txt

echo.
echo Installing package in development mode...
"%COMFYUI_PYTHON%" -m pip install -e .

echo.
echo Installation complete!
echo.
echo You can now restart ComfyUI to see the new CV nodes in the node menu.
echo The nodes will appear under the "CV/" category.
pause
