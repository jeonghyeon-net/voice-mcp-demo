#!/usr/bin/env python3
"""Voice MCP Server - Claude Code용 음성 입출력"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import sounddevice as sd
import torch
import mlx_whisper
from silero_vad import load_silero_vad
from kokoro import KPipeline
from mcp.server.fastmcp import FastMCP

# Silero VAD 로드
torch.set_num_threads(1)
_vad_model = None

def get_vad():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model

mcp = FastMCP("voice")

# 모델 사전 로드
_tts = None
_whisper_loaded = False
_first_load_done = False

def get_tts():
    global _tts
    if _tts is None:
        _tts = KPipeline(lang_code='j', repo_id='hexgrad/Kokoro-82M')
    return _tts

def warmup_whisper():
    """Whisper 모델 사전 로드"""
    global _whisper_loaded
    if not _whisper_loaded:
        mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32),
            path_or_hf_repo="mlx-community/whisper-medium-mlx"
        )
        _whisper_loaded = True

def first_load_notice():
    """첫 로드 시 안내 음성"""
    tts = get_tts()
    for _, _, audio in tts("しょきかちゅう、しばらくおまちください", voice="jf_alpha", speed=1.2):
        if audio is not None:
            sd.play(audio, 24000)
            sd.wait()
            break
    warmup_whisper()
    # VAD 웜업
    vad = get_vad()
    dummy = torch.zeros(512)
    vad(dummy, SAMPLE_RATE)

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# 효과음 미리 생성
def _generate_beep(freq: int, duration: float, volume: float) -> np.ndarray:
    t = np.linspace(0, duration, int(24000 * duration), False)
    tone = np.sin(2 * np.pi * freq * t) * volume
    fade = int(24000 * 0.02)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)

_beep_start_sound = _generate_beep(600, 0.1, 0.4)
_beep_end_sound = _generate_beep(400, 0.08, 0.3)

def beep_start():
    """듣기 시작 효과음"""
    sd.play(_beep_start_sound, 24000)
    sd.wait()
    time.sleep(0.3)

def beep_end():
    """듣기 종료 효과음"""
    sd.play(_beep_end_sound, 24000)
    sd.wait()

# 모델은 첫 사용 시 로드됨 (lazy loading)


@mcp.tool()
def listen(timeout_seconds: int = 300, language: str = "ko") -> str:
    """
    마이크로 음성을 듣고 텍스트로 변환합니다.

    ⚠️ 필수 플로우:
    1. listen() 결과를 받으면
    2. 먼저 speak()로 "~します" 등 할 일을 짧게 말하고
    3. 그 다음 실제 작업 수행
    4. 맥락에 따라 listen() 계속 또는 종료

    예: "ファイルを確認します" → 파일 읽기 → "見つかりました" → ...

    Args:
        timeout_seconds: 최대 대기 시간 (초)
        language: 인식 언어 (ko, en, ja 등)

    Returns:
        인식된 텍스트
    """
    global _first_load_done
    if not _first_load_done:
        first_load_notice()
        _first_load_done = True

    vad_model = get_vad()

    CHUNK_SIZE = 512  # Silero VAD 권장 크기
    MAX_DURATION = 30  # 최대 녹음 30초
    SILENCE_DURATION = 1.5  # 1.5초 침묵 후 종료
    MIN_SPEECH_DURATION = 0.5  # 최소 0.5초 발화해야 유효

    beep_start()  # 🔊 듣기 시작

    audio_buffer = []
    is_speaking = False
    silence_samples = 0
    speech_samples = 0  # 실제 발화 샘플 수
    consecutive_speech = 0  # 연속 음성 프레임
    start_time = time.time()

    captured_audio = None
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, blocksize=CHUNK_SIZE) as stream:
        # 버퍼 비우기
        for _ in range(5):
            stream.read(CHUNK_SIZE)

        while (time.time() - start_time) < timeout_seconds:
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk = chunk.flatten()

            # Silero VAD로 음성 확률 계산
            try:
                chunk_tensor = torch.from_numpy(chunk).float()
                speech_prob = vad_model(chunk_tensor, SAMPLE_RATE).item()
            except Exception as e:
                speech_prob = 0.0

            # 볼륨 체크 (RMS) - 배경 소음 필터링
            rms = np.sqrt(np.mean(chunk ** 2))
            is_voice = speech_prob > 0.85 and rms > 0.02

            if is_voice:
                consecutive_speech += 1
                if not is_speaking and consecutive_speech >= 5:  # 5프레임 연속 음성이어야 시작
                    is_speaking = True
                if is_speaking:
                    audio_buffer.append(chunk)
                    speech_samples += len(chunk)
                silence_samples = 0

                # 최대 길이 체크
                if len(audio_buffer) * CHUNK_SIZE >= MAX_DURATION * SAMPLE_RATE:
                    captured_audio = np.concatenate(audio_buffer)
                    break
            else:
                consecutive_speech = 0  # 연속 음성 리셋

            if not is_voice and is_speaking:
                audio_buffer.append(chunk)
                silence_samples += len(chunk)

                # 최소 발화 시간 충족 + 침묵 지속 시에만 종료
                if speech_samples >= MIN_SPEECH_DURATION * SAMPLE_RATE:
                    if silence_samples >= SILENCE_DURATION * SAMPLE_RATE:
                        captured_audio = np.concatenate(audio_buffer)
                        break
                else:
                    # 발화가 너무 짧으면 리셋 (잡음으로 간주)
                    if silence_samples >= SILENCE_DURATION * SAMPLE_RATE:
                        audio_buffer = []
                        is_speaking = False
                        speech_samples = 0
                        silence_samples = 0

    # 스트림 닫힌 후 처리
    if captured_audio is not None and len(captured_audio) > SAMPLE_RATE * 0.3:
        beep_end()  # 🔊 인식 시작
        result = mlx_whisper.transcribe(
            captured_audio,
            path_or_hf_repo="mlx-community/whisper-medium-mlx",
            language=language
        )
        text = result.get("text", "").strip()

        # 메모리 정리
        del captured_audio
        import gc
        gc.collect()

        if text:
            return f"""[사용자]: {text}

⚠️ 【必須】次のステップ:
1. まずspeak()で「〜します」と短く言う
2. その後、実際の作業を実行
3. 作業完了後もspeak()で結果を伝える

speak()を先に呼ばずに他のツールを使うのは禁止！"""

    return "[타임아웃] 음성이 감지되지 않았습니다."


@mcp.tool()
def listen_fixed(duration_seconds: int = 3, language: str = "ko") -> str:
    """
    지정된 시간 동안 녹음 후 텍스트로 변환합니다.

    Args:
        duration_seconds: 녹음 시간 (초)
        language: 인식 언어 (ko, en, ja 등)

    Returns:
        인식된 텍스트
    """
    audio = sd.rec(int(duration_seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
    sd.wait()
    audio = audio.flatten()

    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo="mlx-community/whisper-medium-mlx",
        language=language
    )
    return result.get("text", "").strip()


@mcp.tool()
def speak(text: str, voice: str = "jf_alpha", speed: float = 1.0) -> str:
    """
    일본어 음성으로 말합니다.

    ⚠️ 다른 도구 호출 전후로 반드시 speak() 호출!
    - Read/Write/Edit 전: "確認します", "作ります", "修正します"
    - Bash 전: "実行します", "テストします"
    - 도구 호출 후: "できました", "見つけました", "エラーです"
    - 여러 작업 시: 각 단계마다 speak() 호출

    2-3단어로 짧게! 예: "次はテストします"

    대화 종료: 모든 작업 완료 후 listen() 생략

    Args:
        text: 일본어 (짧게!)
        voice: 음성
        speed: 속도

    Returns:
        재생 완료
    """
    tts = get_tts()
    for _, _, audio in tts(text, voice=voice, speed=speed):
        if audio is not None:
            sd.play(audio, 24000)
            sd.wait()
            break

    return "→ listen() 호출하세요"


if __name__ == "__main__":
    mcp.run()
