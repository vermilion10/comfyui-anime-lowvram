@echo off
setlocal

rem Keep every cache and temp file on this drive instead of C:
set "ROOT=%~dp0"
set "CACHE=%ROOT%_cache"

set "HF_HOME=%CACHE%\huggingface"
set "TORCH_HOME=%CACHE%\torch"
set "PIP_CACHE_DIR=%CACHE%\pip"
set "XDG_CACHE_HOME=%CACHE%\xdg"
set "TRITON_CACHE_DIR=%CACHE%\triton"
set "YOLO_CONFIG_DIR=%CACHE%\ultralytics"
set "YOLO_OFFLINE=1"
set "CUDA_CACHE_PATH=%CACHE%\cuda"
set "TEMP=%CACHE%\temp"
set "TMP=%CACHE%\temp"

for %%D in ("%HF_HOME%" "%TORCH_HOME%" "%PIP_CACHE_DIR%" "%XDG_CACHE_HOME%" "%TRITON_CACHE_DIR%" "%YOLO_CONFIG_DIR%" "%CUDA_CACHE_PATH%" "%TEMP%") do (
    if not exist "%%~D" mkdir "%%~D"
)

cd /d "%ROOT%ComfyUI_windows_portable"

rem Extra options can be passed through, for example:
rem   "Start ComfyUI.bat" --lowvram
rem   "Start ComfyUI.bat" --fp8_e4m3fn-unet
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --enable-manager --preview-method auto %*

pause
