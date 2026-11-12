@echo off
setlocal
cd /d "%~dp0"

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
