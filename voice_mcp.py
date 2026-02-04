#!/usr/bin/env python3
"""Voice MCP Server - Claude Code용 음성 입출력"""

import warnings
warnings.filterwarnings("ignore")

import re
import time
import numpy as np
import sounddevice as sd
import webrtcvad
import mlx_whisper
import alkana
from kokoro import KPipeline
from mcp.server.fastmcp import FastMCP

# 알파벳 → 카타카나
ALPHA_TO_KANA = {
    'a': 'エー', 'b': 'ビー', 'c': 'シー', 'd': 'ディー', 'e': 'イー',
    'f': 'エフ', 'g': 'ジー', 'h': 'エイチ', 'i': 'アイ', 'j': 'ジェー',
    'k': 'ケー', 'l': 'エル', 'm': 'エム', 'n': 'エヌ', 'o': 'オー',
    'p': 'ピー', 'q': 'キュー', 'r': 'アール', 's': 'エス', 't': 'ティー',
    'u': 'ユー', 'v': 'ブイ', 'w': 'ダブリュー', 'x': 'エックス', 'y': 'ワイ', 'z': 'ゼット'
}

def preprocess_for_tts(text: str) -> str:
    """영어/숫자를 일본어 발음으로 변환"""
    # 숫자 → 일본어
    num_ja = {'0': 'ゼロ', '1': 'いち', '2': 'に', '3': 'さん', '4': 'よん',
              '5': 'ご', '6': 'ろく', '7': 'なな', '8': 'はち', '9': 'きゅう'}
    for num, ja in num_ja.items():
        text = text.replace(num, ja)

    # 영어 단어 → 카타카나
    def replace_english(match):
        word = match.group(0)
        kana = alkana.get_kana(word.lower())
        if kana:
            return kana
        return ''.join(ALPHA_TO_KANA.get(c.lower(), c) for c in word)

    text = re.sub(r'[A-Za-z]+', replace_english, text)
    return text

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
def listen(timeout_seconds: int = 120, language: str = "ko") -> str:
    """
    마이크로 음성을 듣고 텍스트로 변환합니다.

    사용자가 "listen", "듣기", "음성" 등을 입력하면 이 도구를 호출하세요.

    ⚠️ 다른 도구 호출 전후로 speak() 호출 필수. 진행 상황도 수시로 speak().

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

    vad = webrtcvad.Vad(2)
    speech_buffer = []
    silence_count = 0
    is_speaking = False

    SILENCE_THRESHOLD = 20
    MIN_SPEECH_FRAMES = 5
    MAX_SPEECH_FRAMES = 1000  # 약 30초 제한
    max_frames = int(timeout_seconds * SAMPLE_RATE / FRAME_SIZE)
    frame_count = 0

    beep_start()  # 🔊 듣기 시작

    captured_audio = None
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, blocksize=FRAME_SIZE) as stream:
        # 시작 시 버퍼 비우기 (stale 데이터 제거)
        for _ in range(10):
            stream.read(FRAME_SIZE)

        while frame_count < max_frames:
            chunk, _ = stream.read(FRAME_SIZE)
            chunk = chunk.flatten()
            chunk_int16 = (chunk * 32767).astype(np.int16)
            frame_count += 1

            try:
                is_speech = vad.is_speech(chunk_int16.tobytes(), SAMPLE_RATE)
            except:
                is_speech = False

            if is_speech:
                if not is_speaking:
                    is_speaking = True
                speech_buffer.append(chunk)
                silence_count = 0

                # 너무 길면 강제 종료
                if len(speech_buffer) >= MAX_SPEECH_FRAMES:
                    captured_audio = np.concatenate(speech_buffer)
                    break
            elif is_speaking:
                silence_count += 1
                speech_buffer.append(chunk)

                if silence_count >= SILENCE_THRESHOLD:
                    if len(speech_buffer) >= MIN_SPEECH_FRAMES:
                        audio_data = np.concatenate(speech_buffer)
                        rms = np.sqrt(np.mean(audio_data ** 2))

                        if rms >= 0.01:
                            captured_audio = audio_data
                            break  # 스트림 먼저 닫기

                    speech_buffer = []
                    silence_count = 0
                    is_speaking = False

    # 스트림 닫힌 후 처리
    if captured_audio is not None:
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

⚠️ 다른 도구 호출 전후로 speak() 호출 필수. 진행 상황도 수시로 speak()."""

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

    ⚠️ 영어/알파벳 금지! 일본어(한자/히라가나/카타카나)만 사용!

    ⚠️ speak() 후 반드시 listen() 호출! 예외: 사용자가 "끝", "바이바이", "고마워" 등 종료 신호를 보냈을 때만 생략

    Args:
        text: 일본어 텍스트
        voice: 음성
        speed: 속도

    Returns:
        "→ listen() 호출하세요" (종료 신호 없으면)
    """
    tts = get_tts()
    text_processed = preprocess_for_tts(text)
    for _, _, audio in tts(text_processed, voice=voice, speed=speed):
        if audio is not None:
            sd.play(audio, 24000)
            sd.wait()
            break

    return "→ listen() 호출하세요"


if __name__ == "__main__":
    mcp.run()
