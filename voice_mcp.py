#!/usr/bin/env python3
"""Voice MCP Server - 한국어 기본 출력 + 일본어 Kokoro 롤백 지원."""

from __future__ import annotations

import fcntl
import gc
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Protocol

import mlx_whisper
import numpy as np
import sherpa_onnx
import sounddevice as sd
import torch
from kokoro_onnx import Kokoro
from mcp.server.fastmcp import FastMCP
from silero_vad import load_silero_vad

from runtime_config import RuntimeConfig, load_runtime_config

warnings.filterwarnings("ignore")

APP_CONFIG: RuntimeConfig = load_runtime_config()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(APP_CONFIG.log_file, mode="a", encoding="utf-8")],
)
logger = logging.getLogger(__name__)
logger.info("voice 서버 시작 (PID: %s)", os.getpid())
logger.info("출력 언어: %s", APP_CONFIG.output_language)
logger.info("Whisper 모델: %s", APP_CONFIG.whisper_model)

torch.set_num_threads(1)

SAMPLE_RATE = 16000

mcp = FastMCP("voice")

_vad_model = None
_tts_provider: Optional["TTSProvider"] = None
_first_load_done = False
_whisper_loaded = False


class TTSProvider(Protocol):
    language: str

    def synthesize(self, text: str, voice: Optional[str], speed: float) -> tuple[np.ndarray, int]:
        ...


class KoreanSherpaTTS:
    """sherpa-onnx VITS(Mimic3 KSS) 기반 한국어 TTS."""

    language = "ko"

    def __init__(self, model_dir: Path, speaker_id: int):
        model_dir = model_dir.expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(
                f"한국어 TTS 모델 디렉터리를 찾을 수 없습니다: {model_dir}"
            )

        onnx_candidates = sorted(
            p for p in model_dir.glob("*.onnx") if not p.name.endswith(".onnx.json")
        )
        if not onnx_candidates:
            raise FileNotFoundError(f"VITS onnx 모델이 없습니다: {model_dir}")

        tokens_path = model_dir / "tokens.txt"
        espeak_dir = model_dir / "espeak-ng-data"
        if not tokens_path.exists():
            raise FileNotFoundError(f"tokens.txt가 없습니다: {tokens_path}")
        if not espeak_dir.exists():
            raise FileNotFoundError(f"espeak-ng-data 디렉터리가 없습니다: {espeak_dir}")

        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(onnx_candidates[0]),
            tokens=str(tokens_path),
            data_dir=str(espeak_dir),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_config,
            num_threads=2,
            provider="cpu",
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)
        if not tts_config.validate():
            raise RuntimeError("sherpa-onnx TTS 설정 검증에 실패했습니다.")

        self.tts = sherpa_onnx.OfflineTts(tts_config)
        self.speaker_id = speaker_id
        logger.info("한국어 TTS 로드 완료: model=%s", onnx_candidates[0])

    def synthesize(self, text: str, voice: Optional[str], speed: float) -> tuple[np.ndarray, int]:
        generated = self.tts.generate(
            text,
            sid=self.speaker_id,
            speed=max(0.5, min(speed, 2.0)),
        )
        return np.asarray(generated.samples, dtype=np.float32), int(generated.sample_rate)


class JapaneseKokoroTTS:
    """Kokoro ONNX 기반 일본어 TTS."""

    language = "ja"

    def __init__(self, model_path: Path, voices_path: Path):
        model_path = model_path.expanduser().resolve()
        voices_path = voices_path.expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro 모델 파일이 없습니다: {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Kokoro 보이스 파일이 없습니다: {voices_path}")

        self.kokoro = Kokoro(model_path=str(model_path), voices_path=str(voices_path))
        logger.info("일본어 Kokoro 로드 완료: model=%s, voices=%s", model_path, voices_path)

    def synthesize(self, text: str, voice: Optional[str], speed: float) -> tuple[np.ndarray, int]:
        selected_voice = voice or APP_CONFIG.ja_voice
        audio, sample_rate = self.kokoro.create(
            text=text,
            voice=selected_voice,
            speed=max(0.5, min(speed, 2.0)),
            lang="ja",
        )
        return audio.astype(np.float32), int(sample_rate)


def get_vad():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def get_tts_provider() -> TTSProvider:
    global _tts_provider
    if _tts_provider is not None:
        return _tts_provider

    if APP_CONFIG.output_language == "ja":
        _tts_provider = JapaneseKokoroTTS(
            model_path=APP_CONFIG.ja_model_path,
            voices_path=APP_CONFIG.ja_voices_path,
        )
    else:
        _tts_provider = KoreanSherpaTTS(
            model_dir=APP_CONFIG.ko_model_dir,
            speaker_id=APP_CONFIG.ko_speaker_id,
        )
    return _tts_provider


def _play_audio(audio: np.ndarray, sample_rate: int) -> None:
    if audio is None or len(audio) == 0:
        return
    sd.play(audio, sample_rate)
    sd.wait()


def warmup_whisper() -> None:
    global _whisper_loaded
    if _whisper_loaded:
        return
    mlx_whisper.transcribe(
        np.zeros(16000, dtype=np.float32),
        path_or_hf_repo=APP_CONFIG.whisper_model,
    )
    _whisper_loaded = True
    logger.info("Whisper 워밍업 완료")


def first_load_notice() -> None:
    provider = get_tts_provider()
    if provider.language == "ko":
        notice_text = "초기화 중입니다. 잠시만 기다려 주세요."
    else:
        notice_text = "初期化中です。しばらくお待ちください。"

    audio, sr = provider.synthesize(notice_text, voice=None, speed=1.0)
    _play_audio(audio, sr)
    warmup_whisper()

    vad = get_vad()
    dummy = torch.zeros(512)
    vad(dummy, SAMPLE_RATE)


def _generate_beep(freq: int, duration: float, volume: float) -> np.ndarray:
    t = np.linspace(0, duration, int(24000 * duration), False)
    tone = np.sin(2 * np.pi * freq * t) * volume
    fade = int(24000 * 0.02)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)


_beep_start_sound = _generate_beep(600, 0.1, 0.4)
_beep_end_sound = _generate_beep(400, 0.08, 0.3)


def beep_start() -> None:
    sd.play(_beep_start_sound, 24000)
    sd.wait()
    time.sleep(0.3)


def beep_end() -> None:
    sd.play(_beep_end_sound, 24000)
    sd.wait()


@mcp.tool()
def listen(timeout_seconds: int = 300, language: str = "ko") -> str:
    """
    마이크 입력을 텍스트로 변환합니다.

    Args:
        timeout_seconds: 최대 대기 시간(초)
        language: Whisper 입력 언어(ko, en, ja ...)

    Returns:
        인식된 텍스트
    """
    logger.info("=== listen() 시작 (timeout=%s, lang=%s) ===", timeout_seconds, language)

    lock_file = None
    try:
        lock_file = open(APP_CONFIG.lock_file, "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError) as exc:
        logger.warning("마이크 락 획득 실패: %s", exc)
        if lock_file:
            lock_file.close()
        return "[대기] 다른 세션에서 마이크를 사용 중입니다. 잠시 후 다시 시도하세요."

    stream = None
    try:
        global _first_load_done
        if not _first_load_done:
            first_load_notice()
            _first_load_done = True

        vad_model = get_vad()

        chunk_size = 512
        max_duration = 30
        silence_duration = 1.5
        min_speech_duration = 0.5

        beep_start()

        audio_buffer: list[np.ndarray] = []
        lookback_buffer: list[np.ndarray] = []
        lookback_frames = 10
        is_speaking = False
        silence_samples = 0
        speech_samples = 0
        consecutive_speech = 0
        start_time = time.time()
        log_counter = 0
        captured_audio: Optional[np.ndarray] = None

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=chunk_size,
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001
            logger.error("마이크 열기 실패: %s", exc)
            return "[에러] 마이크를 열 수 없습니다. 다른 세션에서 사용 중일 수 있습니다."

        for _ in range(5):
            try:
                stream.read(chunk_size)
            except Exception:
                pass

        while (time.time() - start_time) < timeout_seconds:
            try:
                chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.error("마이크 읽기 에러: %s", exc)
                return "[에러] 마이크 읽기 중 오류가 발생했습니다."

            chunk = chunk.flatten()

            try:
                chunk_tensor = torch.from_numpy(chunk).float()
                speech_prob = vad_model(chunk_tensor, SAMPLE_RATE).item()
            except Exception as exc:  # noqa: BLE001
                logger.error("VAD 에러: %s", exc)
                speech_prob = 0.0

            rms = float(np.sqrt(np.mean(chunk**2)))
            is_voice = speech_prob > 0.85 and rms > 0.02

            log_counter += 1
            if log_counter % 10 == 0:
                logger.debug(
                    "VAD prob=%.3f, rms=%.4f, is_voice=%s, speaking=%s, speech_samples=%s",
                    speech_prob,
                    rms,
                    is_voice,
                    is_speaking,
                    speech_samples,
                )

            if not is_speaking:
                lookback_buffer.append(chunk)
                if len(lookback_buffer) > lookback_frames:
                    lookback_buffer.pop(0)

            if is_voice:
                consecutive_speech += 1
                if not is_speaking and consecutive_speech >= 5:
                    is_speaking = True
                    audio_buffer.extend(lookback_buffer)
                    speech_samples += sum(len(c) for c in lookback_buffer)
                    lookback_buffer = []
                if is_speaking:
                    audio_buffer.append(chunk)
                    speech_samples += len(chunk)
                silence_samples = 0

                if len(audio_buffer) * chunk_size >= max_duration * SAMPLE_RATE:
                    captured_audio = np.concatenate(audio_buffer)
                    break
            else:
                consecutive_speech = 0

            if not is_voice and is_speaking:
                audio_buffer.append(chunk)
                silence_samples += len(chunk)

                if speech_samples >= min_speech_duration * SAMPLE_RATE:
                    if silence_samples >= silence_duration * SAMPLE_RATE:
                        captured_audio = np.concatenate(audio_buffer)
                        break
                elif silence_samples >= silence_duration * SAMPLE_RATE:
                    audio_buffer = []
                    is_speaking = False
                    speech_samples = 0
                    silence_samples = 0

        if captured_audio is not None and len(captured_audio) > SAMPLE_RATE * 0.3:
            beep_end()
            result = mlx_whisper.transcribe(
                captured_audio,
                path_or_hf_repo=APP_CONFIG.whisper_model,
                language=language,
            )
            text = result.get("text", "").strip()
            logger.info("Whisper 완료: %s", text)

            del captured_audio
            gc.collect()

            if text:
                return (
                    f"[사용자]: {text}\n\n"
                    "⚠️ 다음 단계:\n"
                    "1. 먼저 speak()로 짧게 진행 내용을 말하기\n"
                    "2. 실제 작업 실행\n"
                    "3. 작업 완료 후 speak()로 결과 전달\n"
                    "4. 작업이 길어지면 중간 진행상황도 speak()로 공유\n"
                )

        logger.warning("타임아웃 - 음성 감지 실패")
        return "[타임아웃] 음성이 감지되지 않았습니다."

    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                pass


@mcp.tool()
def listen_fixed(duration_seconds: int = 3, language: str = "ko") -> str:
    """
    지정 시간 녹음 후 텍스트로 변환합니다.

    Args:
        duration_seconds: 녹음 시간(초)
        language: Whisper 언어

    Returns:
        인식 결과 텍스트
    """
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
    )
    sd.wait()
    audio = audio.flatten()

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=APP_CONFIG.whisper_model,
        language=language,
    )
    return result.get("text", "").strip()


@mcp.tool()
def speak(text: str, voice: str = "", speed: float = 1.0) -> str:
    """
    현재 설정된 출력 언어로 음성을 재생합니다.

    기본 출력 언어는 runtime.json의 output_language(ko/ja)로 결정됩니다.

    Args:
        text: 재생할 텍스트
        voice: 일본어(Kokoro) 사용 시 보이스명. 한국어(sherpa)에서는 무시됨
        speed: 속도(0.5~2.0)

    Returns:
        listen() 재호출 안내
    """
    try:
        provider = get_tts_provider()
    except Exception as exc:  # noqa: BLE001
        logger.exception("TTS 초기화 실패")
        return f"[에러] TTS 초기화 실패: {exc}"

    try:
        selected_voice: Optional[str]
        if provider.language == "ja":
            selected_voice = voice or APP_CONFIG.ja_voice
        else:
            selected_voice = None
        audio, sample_rate = provider.synthesize(
            text=text,
            voice=selected_voice,
            speed=speed,
        )
        _play_audio(audio, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.exception("TTS 재생 실패")
        return f"[에러] 음성 합성 실패: {exc}"

    return "→ listen() 호출하세요"


def run_server() -> None:
    mcp.run()


if __name__ == "__main__":
    run_server()

