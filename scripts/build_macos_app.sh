#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 실행 파일을 찾을 수 없습니다: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/5] 의존성 설치"
"$PYTHON_BIN" -m pip install -r requirements.txt pyinstaller

echo "[2/5] 모델 준비 (assets/models)"
"$PYTHON_BIN" setup_models.py

echo "[3/5] 이전 빌드 정리"
rm -rf build dist

echo "[4/5] MCP 서버 바이너리 빌드"
"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name voice-mcp-server \
  --collect-submodules mcp \
  --hidden-import mcp.server.fastmcp \
  --hidden-import kokoro_onnx \
  --hidden-import sherpa_onnx \
  --add-data "assets/models:assets/models" \
  --add-data "runtime_config.py:." \
  voice_mcp_server.py

echo "[5/5] 런처 앱 빌드 (.app)"
"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name VoiceMCPLauncher \
  --add-data "dist/voice-mcp-server:voice-mcp-server" \
  --add-data "runtime_config.py:." \
  app_main.py

SERVER_IN_APP="dist/VoiceMCPLauncher.app/Contents/Resources/voice-mcp-server/voice-mcp-server"
if [[ -f "$SERVER_IN_APP" ]]; then
  chmod +x "$SERVER_IN_APP"
fi

echo
echo "빌드 완료"
echo "앱 경로: $ROOT_DIR/dist/VoiceMCPLauncher.app"
echo "MCP 서버 경로: $ROOT_DIR/$SERVER_IN_APP"

