#!/usr/bin/env python3
"""Voice MCP 런타임 설정/경로 유틸리티."""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_NAME = "VoiceMCP"

DEFAULT_CONFIG: dict[str, Any] = {
    "output_language": "ko",
    "whisper": {
        "model_path": "assets/models/whisper-medium-mlx",
        "fallback_repo": "mlx-community/whisper-medium-mlx",
    },
    "ko_tts": {
        "model_dir": "assets/models/vits-mimic3-ko_KO-kss_low",
        "speaker_id": 0,
        "speed": 1.0,
    },
    "ja_tts": {
        "model_path": "assets/models/kokoro/kokoro-v1.0.onnx",
        "voices_path": "assets/models/kokoro/voices-v1.0.bin",
        "voice": "jf_alpha",
        "speed": 1.0,
    },
}


@dataclass
class RuntimeConfig:
    output_language: str
    whisper_model: str
    ko_model_dir: Path
    ko_speaker_id: int
    ko_speed: float
    ja_model_path: Path
    ja_voices_path: Path
    ja_voice: str
    ja_speed: float
    app_support_dir: Path
    log_file: Path
    lock_file: Path
    config_file: Path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_NAME


def get_runtime_config_path() -> Path:
    return get_app_support_dir() / "runtime.json"


def _resource_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
        exe = Path(sys.executable).resolve()
        roots.append(exe.parent)
        # .app/Contents/Resources
        roots.append(exe.parent.parent / "Resources")

    roots.append(Path(__file__).resolve().parent)
    roots.append(Path.cwd())

    unique: list[Path] = []
    seen: set[str] = set()
    for path in roots:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _resolve_existing_path(raw_path: str, fallback_candidates: list[str] | None = None) -> Path | None:
    candidates = [raw_path]
    if fallback_candidates:
        candidates.extend(fallback_candidates)

    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path
        for root in _resource_roots():
            resolved = (root / candidate_path).resolve()
            if resolved.exists():
                return resolved
    return None


def _ensure_runtime_file(config_file: Path) -> dict[str, Any]:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if not config_file.exists():
        config_file.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return copy.deepcopy(DEFAULT_CONFIG)

    try:
        loaded = json.loads(config_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return copy.deepcopy(DEFAULT_CONFIG)
        return _deep_merge(DEFAULT_CONFIG, loaded)
    except Exception:
        return copy.deepcopy(DEFAULT_CONFIG)


def load_runtime_config() -> RuntimeConfig:
    config_file = get_runtime_config_path()
    config = _ensure_runtime_file(config_file)

    output_language = str(
        os.environ.get("VOICE_OUTPUT_LANGUAGE", config.get("output_language", "ko"))
    ).lower()
    if output_language not in {"ko", "ja"}:
        output_language = "ko"

    whisper_cfg = config.get("whisper", {})
    whisper_override = os.environ.get("VOICE_WHISPER_MODEL_PATH", "").strip()
    whisper_raw = whisper_override or str(
        whisper_cfg.get("model_path", DEFAULT_CONFIG["whisper"]["model_path"])
    )
    whisper_fallback = str(
        whisper_cfg.get("fallback_repo", DEFAULT_CONFIG["whisper"]["fallback_repo"])
    )
    whisper_model_path = _resolve_existing_path(whisper_raw)
    whisper_model = str(whisper_model_path) if whisper_model_path else whisper_fallback

    ko_cfg = config.get("ko_tts", {})
    ko_model_override = os.environ.get("VOICE_KO_MODEL_DIR", "").strip()
    ko_model_raw = ko_model_override or str(
        ko_cfg.get("model_dir", DEFAULT_CONFIG["ko_tts"]["model_dir"])
    )
    ko_model_dir = _resolve_existing_path(ko_model_raw)
    if ko_model_dir is None:
        # 빌드 전 개발환경에서도 즉시 실패 원인 확인 가능하도록 경로를 보존
        ko_model_dir = Path(ko_model_raw)

    ko_speaker_id = int(
        os.environ.get("VOICE_KO_SPEAKER_ID", ko_cfg.get("speaker_id", 0))
    )
    ko_speed = float(os.environ.get("VOICE_KO_SPEED", ko_cfg.get("speed", 1.0)))

    ja_cfg = config.get("ja_tts", {})
    ja_model_override = os.environ.get("VOICE_JA_MODEL_PATH", "").strip()
    ja_voices_override = os.environ.get("VOICE_JA_VOICES_PATH", "").strip()
    ja_voice = os.environ.get("VOICE_JA_VOICE", ja_cfg.get("voice", "jf_alpha"))
    ja_speed = float(os.environ.get("VOICE_JA_SPEED", ja_cfg.get("speed", 1.0)))

    ja_model_path = _resolve_existing_path(
        ja_model_override or str(ja_cfg.get("model_path", DEFAULT_CONFIG["ja_tts"]["model_path"])),
        fallback_candidates=["kokoro-v1.0.onnx"],
    )
    ja_voices_path = _resolve_existing_path(
        ja_voices_override or str(ja_cfg.get("voices_path", DEFAULT_CONFIG["ja_tts"]["voices_path"])),
        fallback_candidates=["voices-v1.0.bin"],
    )

    if ja_model_path is None:
        ja_model_path = Path("kokoro-v1.0.onnx")
    if ja_voices_path is None:
        ja_voices_path = Path("voices-v1.0.bin")

    app_support_dir = get_app_support_dir()
    app_support_dir.mkdir(parents=True, exist_ok=True)

    return RuntimeConfig(
        output_language=output_language,
        whisper_model=whisper_model,
        ko_model_dir=ko_model_dir,
        ko_speaker_id=ko_speaker_id,
        ko_speed=ko_speed,
        ja_model_path=ja_model_path,
        ja_voices_path=ja_voices_path,
        ja_voice=str(ja_voice),
        ja_speed=ja_speed,
        app_support_dir=app_support_dir,
        log_file=app_support_dir / "voice_debug.log",
        lock_file=app_support_dir / ".mic_lock",
        config_file=config_file,
    )


def save_output_language(language: str) -> None:
    language = language.lower().strip()
    if language not in {"ko", "ja"}:
        raise ValueError("language must be 'ko' or 'ja'")

    config_path = get_runtime_config_path()
    config = _ensure_runtime_file(config_path)
    config["output_language"] = language
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

