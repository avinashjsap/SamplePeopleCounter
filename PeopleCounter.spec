# PyInstaller bundle definition. Build this on Windows for Windows output.
from PyInstaller.utils.hooks import collect_all

ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all("ultralytics")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=ultralytics_binaries,
    datas=ultralytics_datas + [("yolo26n.pt", ".")],
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
