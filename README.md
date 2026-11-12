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

### Live camera classification

To show classifier labels next to each tracked person, place `gender_classification.pt` beside `main.py` and keep the age models in `face_detector/` and `age_detector/`. The app combines each person's ID, detection confidence, gender label, and estimated age range in one Ultralytics-style tag. Age is classified into broad brackets (`0-2`, `4-6`, `8-12`, `15-20`, `25-32`, `38-43`, `48-53`, `60-100`), not an exact age. The estimate is based on visible facial appearance and can be inaccurate, especially when faces are small or obscured. To include all models in a Windows build, keep these files alongside `PeopleCounter.spec` when running `build_windows.bat`.

The age model follows the [PyImageSearch OpenCV age-detection walkthrough](https://pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/) and uses the face detector from [OpenCV's DNN samples](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) and age weights from the [original Levi and Hassner repository](https://github.com/GilLevi/AgeGenderDeepLearning).

### Tracking log

While the camera runs, the app prints classification updates to the console with timestamps and log levels. CSV logging is disabled for now; set `CSV_LOGGING_ENABLED = True` in `main.py` to re-enable it.

## Apple Silicon DMG

Build the `.dmg` on an Apple Silicon Mac with the project dependencies installed in `.venv`:

```bash
./build_macos_dmg.sh
```

The script creates `dist/PeopleCounter-AppleSilicon-<timestamp>.dmg`. Open the DMG and double-click `Run People Counter.command` to launch the app with console output visible in Terminal. The bundled `.app` is also included. The build bundles the YOLO, gender, face-detection, and age-estimation model files.

