#!/usr/bin/env python3
import sounddevice as sd
import numpy as np

print("마이크 테스트 (3초간 녹음)")
print("사용 가능한 장치:")
print(sd.query_devices())
print("\n기본 입력:", sd.default.device[0])
print("\n녹음 시작...")

duration = 3
audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype=np.float32)
sd.wait()

rms = np.sqrt(np.mean(audio ** 2))
peak = np.max(np.abs(audio))

print(f"\n결과:")
print(f"  RMS: {rms:.4f}")
print(f"  Peak: {peak:.4f}")

if peak < 0.001:
    print("\n⚠️  입력이 거의 없음 - 마이크 확인 필요")
elif peak < 0.01:
    print("\n⚠️  입력이 매우 작음 - 볼륨 높이거나 마이크 가까이")
else:
    print("\n✓ 입력 정상")
