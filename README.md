# People Counter for Windows

This app counts people crossing the center line of the laptop webcam view. Press **X** in the video window to quit.

## Build the Windows executable

Build on a Windows 10 or Windows 11 **64-bit** computer. PyInstaller does not create Windows executables from macOS. Install Python 3.10 or newer from python.org and make sure the Python launcher (`py`) is available.

1. Copy this project folder to the Windows computer. Keep `main.py`, `yolo26n.pt`, `requirements.txt`, `PeopleCounter.spec`, and `build_windows.bat` together.
2. Double-click `build_windows.bat` and wait for it to finish. The first build downloads Python packages and can take several minutes.
3. Copy the entire `dist\PeopleCounter` folder to any target Windows 10/11 64-bit laptop. Start `PeopleCounter.exe` inside it.

The target laptop does not need Python installed or an internet connection. It needs a webcam and enough disk space for the bundled application and model. Windows may ask for camera permission the first time the app runs.

## Notes

- The output is a folder bundle, not a standalone single-file executable. Keep all files under `dist\PeopleCounter` together when copying or sharing it.
- `yolo26n.pt` is bundled with the app, so the model does not need to download at startup.
- The app currently uses the default camera (device 0). Press **X** while the video window is focused to exit.
- A build made on Windows targets Windows. Rebuild after changing the Python code, dependencies, or model.
