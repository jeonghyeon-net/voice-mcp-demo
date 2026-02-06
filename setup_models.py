#!/usr/bin/env python3
"""앱 빌드용 모델 준비 스크립트 (최종 사용자 실행용 아님)."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

import mlx_whisper
import numpy as np
import sherpa_onnx
from huggingface_hub import snapshot_download
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent
ASSETS_MODELS = ROOT / "assets" / "models"

WHISPER_REPO = "mlx-community/whisper-medium-mlx"
WHISPER_DIR = ASSETS_MODELS / "whisper-medium-mlx"

KO_TTS_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "vits-mimic3-ko_KO-kss_low.tar.bz2"
)
KO_TTS_DIR = ASSETS_MODELS / "vits-mimic3-ko_KO-kss_low"
KO_TTS_ARCHIVE = ASSETS_MODELS / "vits-mimic3-ko_KO-kss_low.tar.bz2"

KOKORO_SRC_MODEL = ROOT / "kokoro-v1.0.onnx"
KOKORO_SRC_VOICES = ROOT / "voices-v1.0.bin"
KOKORO_DST_DIR = ASSETS_MODELS / "kokoro"


def ensure_dirs() -> None:
    ASSETS_MODELS.mkdir(parents=True, exist_ok=True)


def prepare_whisper() -> None:
    print(f"[1/3] Whisper 모델 준비: {WHISPER_REPO}")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=WHISPER_REPO,
        local_dir=str(WHISPER_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    mlx_whisper.transcribe(
        np.zeros(16000, dtype=np.float32),
        path_or_hf_repo=str(WHISPER_DIR),
    )
    print(f"  ✓ 완료: {WHISPER_DIR}")


def prepare_korean_tts() -> None:
    print("[2/3] 한국어 TTS 모델 준비: sherpa-onnx vits-mimic3-ko_KO-kss_low")
    if not KO_TTS_DIR.exists():
        urllib.request.urlretrieve(KO_TTS_URL, KO_TTS_ARCHIVE)  # noqa: S310
        with tarfile.open(KO_TTS_ARCHIVE, "r:bz2") as tar:
            tar.extractall(path=ASSETS_MODELS)
        if KO_TTS_ARCHIVE.exists():
            KO_TTS_ARCHIVE.unlink()

    onnx_candidates = sorted(KO_TTS_DIR.glob("*.onnx"))
    if not onnx_candidates:
        raise FileNotFoundError(f"onnx 파일을 찾을 수 없습니다: {KO_TTS_DIR}")
    tokens = KO_TTS_DIR / "tokens.txt"
    espeak = KO_TTS_DIR / "espeak-ng-data"
    if not tokens.exists() or not espeak.exists():
        raise FileNotFoundError("한국어 TTS 모델 구성 파일이 누락되었습니다.")

    vits_cfg = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(onnx_candidates[0]),
        tokens=str(tokens),
        data_dir=str(espeak),
    )
    model_cfg = sherpa_onnx.OfflineTtsModelConfig(vits=vits_cfg, provider="cpu", num_threads=2)
    tts_cfg = sherpa_onnx.OfflineTtsConfig(model=model_cfg)
    if not tts_cfg.validate():
        raise RuntimeError("한국어 TTS 모델 검증 실패")
    tts = sherpa_onnx.OfflineTts(tts_cfg)
    _ = tts.generate("안녕하세요.", sid=0, speed=1.0)
    print(f"  ✓ 완료: {KO_TTS_DIR}")


def prepare_kokoro() -> None:
    print("[3/3] 일본어 롤백용 Kokoro 모델 준비")
    if not KOKORO_SRC_MODEL.exists() or not KOKORO_SRC_VOICES.exists():
        raise FileNotFoundError(
            "루트 경로의 Kokoro 파일이 필요합니다: "
            f"{KOKORO_SRC_MODEL}, {KOKORO_SRC_VOICES}"
        )

    KOKORO_DST_DIR.mkdir(parents=True, exist_ok=True)
    model_dst = KOKORO_DST_DIR / "kokoro-v1.0.onnx"
    voices_dst = KOKORO_DST_DIR / "voices-v1.0.bin"
    shutil.copy2(KOKORO_SRC_MODEL, model_dst)
    shutil.copy2(KOKORO_SRC_VOICES, voices_dst)

    kokoro = Kokoro(model_path=str(model_dst), voices_path=str(voices_dst))
    _audio, _sr = kokoro.create("テストです。", voice="jf_alpha", lang="ja")
    print(f"  ✓ 완료: {KOKORO_DST_DIR}")


def main() -> None:
    print("=" * 60)
    print("Voice MCP 빌드용 모델 준비")
    print("=" * 60)
    print("주의: 이 스크립트는 개발자 빌드 단계용입니다.")

    ensure_dirs()
    prepare_whisper()
    prepare_korean_tts()
    prepare_kokoro()

    print("\n모델 준비가 완료되었습니다.")
    print(f"모델 위치: {ASSETS_MODELS}")


if __name__ == "__main__":
    main()

