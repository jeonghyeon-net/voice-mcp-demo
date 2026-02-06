#!/usr/bin/env python3
"""모델 사전 다운로드 스크립트 - Claude Code 사용 전 한 번 실행"""

import sys
import subprocess
from pathlib import Path


def ensure_kokoro_ja_runtime() -> None:
    """Kokoro 일본어 파이프라인 필수 의존성 점검/보정."""
    try:
        import pyopenjtalk  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "pyopenjtalk가 필요합니다. `python -m pip install \"misaki[ja]\" pyopenjtalk` 실행 후 다시 시도하세요."
        ) from e

    try:
        import unidic
    except Exception as e:
        raise RuntimeError(
            "unidic가 필요합니다. `python -m pip install \"misaki[ja]\" pyopenjtalk` 실행 후 다시 시도하세요."
        ) from e

    dicdir = Path(getattr(unidic, "DICDIR", ""))
    mecabrc = dicdir / "mecabrc"
    if mecabrc.exists():
        return

    print("  - unidic 사전 다운로드 중...")
    subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)

    # 다운로드 후 경로 재검증
    import importlib
    unidic = importlib.import_module("unidic")
    dicdir = Path(getattr(unidic, "DICDIR", ""))
    mecabrc = dicdir / "mecabrc"
    if not mecabrc.exists():
        raise RuntimeError(
            f"unidic 사전 초기화 실패: {mecabrc} 파일이 없습니다."
        )

print("=" * 50)
print("Voice MCP 모델 다운로드")
print("=" * 50)

# 1. Silero VAD
print("\n[1/3] Silero VAD 다운로드 중...")
import torch
torch.set_num_threads(1)
from silero_vad import load_silero_vad
vad = load_silero_vad()
print("✓ Silero VAD 완료")

# 2. MLX Whisper
print("\n[2/3] MLX Whisper 다운로드 중...")
import numpy as np
import mlx_whisper
mlx_whisper.transcribe(
    np.zeros(16000, dtype=np.float32),
    path_or_hf_repo="mlx-community/whisper-medium-mlx"
)
print("✓ MLX Whisper 완료")

# 3. Kokoro TTS
print("\n[3/3] Kokoro TTS 다운로드 중...")
ensure_kokoro_ja_runtime()
from kokoro import KPipeline
tts = KPipeline(lang_code='j', repo_id='hexgrad/Kokoro-82M')
# 테스트 생성
for _, _, audio in tts("テスト", voice="jf_alpha", speed=1.0):
    break
print("✓ Kokoro TTS 완료")

print("\n" + "=" * 50)
print("모든 모델 다운로드 완료!")
print("이제 Claude Code에서 voice MCP를 사용할 수 있습니다.")
print("=" * 50)
