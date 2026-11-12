#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
export PYINSTALLER_CONFIG_DIR="$PROJECT_DIR/build/pyinstaller-config"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Build this DMG on an Apple Silicon Mac (arm64)."
  exit 1
fi

required_files=(
  "yolo26n.pt"
  "gender_classification.pt"
  "face_detector/deploy.prototxt"
  "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
  "age_detector/age_deploy.prototxt"
  "age_detector/age_net.caffemodel"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required model file: $required_file"
    exit 1
  fi
done

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=".build-macos-venv/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    python3 -m venv .build-macos-venv
  fi
  .build-macos-venv/bin/python -m pip install -r requirements.txt
fi

"$PYTHON_BIN" -c 'import cv2, torch, ultralytics, PyInstaller'
"$PYTHON_BIN" -m PyInstaller --noconfirm --clean \
  --distpath dist/macos --workpath build/macos PeopleCounter_mac.spec

BUILD_STAMP="$(date +%Y%m%d-%H%M%S)"
DMG_PATH="dist/PeopleCounter-AppleSilicon-${BUILD_STAMP}.dmg"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/peoplecounter-dmg.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT

cp -R dist/macos/PeopleCounter.app "$STAGING_DIR/PeopleCounter.app"
cat > "$STAGING_DIR/Run People Counter.command" <<'EOF'
#!/bin/bash
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$APP_DIR/PeopleCounter.app/Contents/MacOS/PeopleCounter"
EOF
chmod +x "$STAGING_DIR/Run People Counter.command"

hdiutil create -volname "People Counter" -srcfolder "$STAGING_DIR" \
  -ov -format UDZO "$DMG_PATH"

echo "DMG created: $DMG_PATH"
echo "Run People Counter.command from the mounted DMG to see console logs in Terminal."
