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

_beep_start_sound = _generate_beep(880, 0.25, 0.7)
_beep_end_sound = _generate_beep(440, 0.15, 0.6)

def beep_start():
    """듣기 시작 효과음"""
    sd.play(_beep_start_sound, 24000)
    sd.wait()

def beep_end():
    """듣기 종료 효과음"""
    sd.play(_beep_end_sound, 24000)
    sd.wait()

# 서버 시작 시 모델 로드 & 효과음 테스트
warmup_whisper()
get_tts()
beep_start()


@mcp.tool()
def listen(timeout_seconds: int = 1800, language: str = "ko") -> str:
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
    vad = webrtcvad.Vad(2)
    speech_buffer = []
    silence_count = 0
    is_speaking = False

    SILENCE_THRESHOLD = 20
    MIN_SPEECH_FRAMES = 5
    max_frames = int(timeout_seconds * SAMPLE_RATE / FRAME_SIZE)
    frame_count = 0

    beep_start()  # 🔊 듣기 시작

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, blocksize=FRAME_SIZE) as stream:
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
            elif is_speaking:
                silence_count += 1
                speech_buffer.append(chunk)

                if silence_count >= SILENCE_THRESHOLD:
                    if len(speech_buffer) >= MIN_SPEECH_FRAMES:
                        audio_data = np.concatenate(speech_buffer)
                        rms = np.sqrt(np.mean(audio_data ** 2))

                        if rms >= 0.01:
                            beep_end()  # 🔊 인식 시작
                            result = mlx_whisper.transcribe(
                                audio_data,
                                path_or_hf_repo="mlx-community/whisper-medium-mlx",
                                language=language
                            )
                            text = result.get("text", "").strip()
                            if text:
                                return f"""[사용자]: {text}

⚠️ 다른 도구 호출 전후로 speak() 호출 필수. 진행 상황도 수시로 speak()."""

                    speech_buffer = []
                    silence_count = 0
                    is_speaking = False

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
    text_processed = preprocess_for_tts(text)
    for _, _, audio in tts(text_processed, voice=voice, speed=speed):
        if audio is not None:
            sd.play(audio, 24000)
            sd.wait()
            break

    return "再生完了"


if __name__ == "__main__":
    mcp.run()
