@echo off
setlocal
cd /d "%~dp0"

if not exist "yolo26n.pt" goto :missing_model
if not exist "gender_classification.pt" goto :missing_model
if not exist "face_detector\deploy.prototxt" goto :missing_model
if not exist "face_detector\res10_300x300_ssd_iter_140000.caffemodel" goto :missing_model
if not exist "age_detector\age_deploy.prototxt" goto :missing_model
if not exist "age_detector\age_net.caffemodel" goto :missing_model

py -3 -m venv .build-venv
if errorlevel 1 goto :error
call .build-venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python -m PyInstaller --noconfirm --clean PeopleCounter.spec
if errorlevel 1 goto :error

echo.
echo Build complete: dist\PeopleCounter\PeopleCounter.exe
echo Copy the entire dist\PeopleCounter folder to the target Windows laptop.
exit /b 0

:error
echo Build failed. See the error above.
exit /b 1

:missing_model
echo Build failed. One or more required model files are missing.
echo Required: yolo26n.pt, gender_classification.pt, face_detector\*, and age_detector\*.
exit /b 1
