#!/usr/bin/env python3
"""음성 따라말하기 - MLX Whisper + Kokoro + Ollama Cloud"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import sounddevice as sd
import webrtcvad
import mlx_whisper
from kokoro import KPipeline
from ollama import Client

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SILENCE_THRESHOLD = 20
MIN_SPEECH_FRAMES = 5

# Whisper 환각 필터
HALLUCINATIONS = {"celebrated", "thank you", "thanks for watching", "subscribe", "bye", "goodbye"}

import re
import alkana

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
        # alkana로 먼저 시도
        kana = alkana.get_kana(word.lower())
        if kana:
            return kana
        # 실패하면 알파벳 하나씩 변환
        return ''.join(ALPHA_TO_KANA.get(c.lower(), c) for c in word)

    text = re.sub(r'[A-Za-z]+', replace_english, text)
    return text


SYSTEM_PROMPT = "日本語のみで応答せよ。英語・外来語は必ずカタカナ表記。数字は漢数字か読み仮名で書け。韓国語・中国語・アルファベット禁止。入力は韓国語で誤字あり。短く。"


def generate_response(client: Client, model: str, user_text: str) -> str:
    """LLM으로 일본어 응답 생성"""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "すみません、エラーが発生しました。"


def is_hallucination(text: str) -> bool:
    t = text.lower().strip()
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    for h in HALLUCINATIONS:
        if h in t:
            return True
    return False


def main():
    print("=" * 30)
    print("Voice Echo")
    print("=" * 30)

    print("[1/3] Whisper (MLX)...", end=" ", flush=True)
    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo="mlx-community/whisper-medium-mlx")
    print("OK (Apple GPU)")

    print("[2/3] Kokoro TTS...", end=" ", flush=True)
    tts = KPipeline(lang_code='j', repo_id='hexgrad/Kokoro-82M')
    print("OK")

    print("[3/3] Ollama Cloud...", end=" ", flush=True)
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print("ERROR: OLLAMA_API_KEY 환경변수 필요")
        return
    llm = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    llm_model = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
    print(f"OK ({llm_model})")

    vad = webrtcvad.Vad(2)
    speech_buffer = []
    silence_count = 0
    is_speaking = False

    print("\n🎤 Ready\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, blocksize=FRAME_SIZE) as stream:
        while True:
            try:
                chunk, _ = stream.read(FRAME_SIZE)
                chunk = chunk.flatten()
                chunk_int16 = (chunk * 32767).astype(np.int16)

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
                            audio_float = audio_data.astype(np.float32)
                            rms = np.sqrt(np.mean(audio_float ** 2))

                            if rms >= 0.01:
                                result = mlx_whisper.transcribe(
                                    audio_float,
                                    path_or_hf_repo="mlx-community/whisper-medium-mlx",
                                    language="ko"
                                )
                                text = result.get("text", "").strip()

                                if text and len(text) > 1 and not is_hallucination(text):
                                    print(f"🎤 {text}")

                                    response = generate_response(llm, llm_model, text)
                                    response_tts = preprocess_for_tts(response)
                                    print(f"🔊 {response}")
                                    for _, _, audio in tts(response_tts, voice='jf_alpha', speed=1.0):
                                        if audio is not None:
                                            sd.play(audio, 24000)
                                            sd.wait()
                                            break

                        speech_buffer = []
                        silence_count = 0
                        is_speaking = False

            except KeyboardInterrupt:
                print("\nBye")
                break


if __name__ == "__main__":
    main()
