#!/usr/bin/env python3
"""Silero VAD + RMS 테스트"""

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad
import time

torch.set_num_threads(1)
print("VAD 로딩...")
vad = load_silero_vad()

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
VAD_THRESHOLD = 0.85
RMS_THRESHOLD = 0.015

print(f"테스트 시작 - VAD>{VAD_THRESHOLD} AND RMS>{RMS_THRESHOLD} (10초간)")
print("-" * 50)

start = time.time()
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, blocksize=CHUNK_SIZE) as stream:
    while time.time() - start < 10:
        chunk, _ = stream.read(CHUNK_SIZE)
        chunk = chunk.flatten()

        try:
            chunk_tensor = torch.from_numpy(chunk).float()
            prob = vad(chunk_tensor, SAMPLE_RATE).item()
            rms = np.sqrt(np.mean(chunk ** 2))

            is_voice = prob > VAD_THRESHOLD and rms > RMS_THRESHOLD

            if is_voice:
                print(f"VAD:{prob:.2f} RMS:{rms:.3f} ← 음성!")
            elif prob > 0.5 or rms > 0.01:
                print(f"VAD:{prob:.2f} RMS:{rms:.3f}")
        except Exception as e:
            print(f"에러: {e}")

print("-" * 50)
print("테스트 완료")
