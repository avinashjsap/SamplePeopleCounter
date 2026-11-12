# PyInstaller bundle definition for an Apple Silicon macOS app.
from PyInstaller.utils.hooks import collect_all
import os

ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all("ultralytics")
model_files = [("yolo26n.pt", "."), ("gender_classification.pt", ".")]
required_model_files = (
    "yolo26n.pt",
    "gender_classification.pt",
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    "age_detector/age_deploy.prototxt",
    "age_detector/age_net.caffemodel",
)
for model_file in required_model_files:
    if not os.path.isfile(model_file):
        raise FileNotFoundError(model_file)
    if model_file not in ("yolo26n.pt", "gender_classification.pt"):
        model_files.append((model_file, os.path.dirname(model_file)))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=ultralytics_binaries,
    datas=ultralytics_datas + model_files,
    hiddenimports=ultralytics_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PeopleCounter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PeopleCounter",
)
app = BUNDLE(
    coll,
    name="PeopleCounter.app",
    icon=None,
    bundle_identifier="com.peopletracker.peoplecounter",
    info_plist={
        "NSCameraUsageDescription": "People Counter uses the camera to count people and show visual model estimates.",
    },
)
