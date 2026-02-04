# Voice MCP Server

Claude Code에서 음성으로 대화할 수 있게 해주는 MCP 서버입니다.

- **STT**: MLX Whisper (Apple Silicon 최적화)
- **TTS**: Kokoro (다국어 지원: 일본어, 영어, 한국어, 중국어 등)
- **VAD**: Silero VAD + RMS 이중 필터

## 기본 설정

> ⚠️ 현재 코드는 **한국어 입력 → 일본어 출력**으로 하드코딩되어 있습니다.
> 다른 언어로 변경하려면 [언어 변경 가이드](#언어-변경-가이드) 섹션을 참고하세요.

## 요구사항

- macOS (Apple Silicon M1/M2/M3)
- Python 3.11 이상
- Claude Code CLI
- 마이크 (맥북 내장 마이크 또는 외장)

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/jeonghyeon-net/vtuber.git
cd vtuber
```

### 2. Python 3.11 설치 (없는 경우)

```bash
brew install python@3.11
```

### 3. 가상환경 생성 및 활성화

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 4. 의존성 설치

```bash
pip install -r requirements.txt
```

### 5. 모델 다운로드

```bash
python setup_models.py
```

> ⚠️ **필수**: Claude Code 사용 전 반드시 실행하세요.
> Whisper, Kokoro TTS, Silero VAD 모델을 미리 다운로드합니다.
> 첫 실행 시 약 2-3GB 다운로드됩니다.

### 6. 설치 확인

```bash
# VAD 테스트 (마이크 테스트)
python test_vad.py
```

말하면 음성 확률이 표시되어야 합니다.

## MCP 설정

`~/.mcp.json` 파일을 생성하거나 수정:

```json
{
  "mcpServers": {
    "voice": {
      "command": "/경로/vtuber/venv/bin/python",
      "args": ["/경로/vtuber/voice_mcp.py"]
    }
  }
}
```

> `/경로/`를 실제 경로로 변경하세요.

## 사용법

Claude Code에서:

```
> listen
```

입력하면 음성 인식 모드가 시작됩니다.

### 도구

| 도구 | 설명 |
|------|------|
| `listen()` | 마이크로 음성 듣기 (한국어) |
| `speak(text)` | 일본어 TTS로 응답 |
| `listen_fixed(duration)` | 고정 시간 녹음 |

### 플로우

1. `listen` 입력 → 비프음 후 말하기
2. Claude가 일본어로 응답 (`speak`)
3. 대화 계속 또는 종료

## 설정값

`voice_mcp.py`에서 조정 가능:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `VAD_THRESHOLD` | 0.85 | 음성 감지 임계값 |
| `RMS_THRESHOLD` | 0.015 | 볼륨 임계값 |
| `SILENCE_DURATION` | 1.5초 | 침묵 후 종료 시간 |
| `timeout_seconds` | 300초 | 최대 대기 시간 |

## 문제 해결

### 음성이 인식 안 됨
- 마이크 권한 확인
- RMS_THRESHOLD 낮추기 (0.01)

### 배경 소음에 반응함
- VAD_THRESHOLD 높이기 (0.9)
- RMS_THRESHOLD 높이기 (0.03)

### MCP 연결 실패
- Python 경로 확인
- `python -m py_compile voice_mcp.py`로 문법 검사

## 테스트

```bash
# VAD 테스트
./venv/bin/python test_vad.py

# 독립 실행 (echo.py)
./run.sh
```

## 라이선스

MIT

## 언어 변경 가이드

기본값은 **한국어 입력 → 일본어 출력**입니다. 다른 언어로 변경하려면:

### 영어 TTS로 변경

`voice_mcp.py` 수정:

```python
# 1. TTS 언어 코드 변경 (get_tts 함수)
_tts = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
# 'a' = 미국 영어, 'b' = 영국 영어, 'j' = 일본어
# 'k' = 한국어, 'z' = 중국어, 'f' = 프랑스어 등

# 2. 음성 변경 (speak 함수의 기본값)
def speak(text: str, voice: str = "af_heart", speed: float = 1.0) -> str:
# 영어 음성: af_heart, af_bella, am_adam, am_michael 등
```

### Kokoro 지원 언어

| 코드 | 언어 |
|------|------|
| `a` | 미국 영어 |
| `b` | 영국 영어 |
| `j` | 일본어 |
| `k` | 한국어 |
| `z` | 중국어 |
| `f` | 프랑스어 |
| `e` | 스페인어 |
| `i` | 이탈리아어 |
| `p` | 포르투갈어 |
| `h` | 힌디어 |

### 영어 음성 목록

| 음성 | 설명 |
|------|------|
| `af_heart` | 미국 여성 (기본 추천) |
| `af_bella` | 미국 여성 |
| `af_sarah` | 미국 여성 |
| `am_adam` | 미국 남성 |
| `am_michael` | 미국 남성 |
| `bf_emma` | 영국 여성 |
| `bm_george` | 영국 남성 |

### speak() 프롬프트 변경

Claude가 영어로 응답하도록 `speak` 함수의 docstring 수정:

```python
@mcp.tool()
def speak(text: str, voice: str = "af_heart", speed: float = 1.0) -> str:
    """
    Speak in English.

    ⚠️ Text must be in English only!

    Args:
        text: English text
        voice: Voice
        speed: Speed

    Returns:
        Playback complete
    """
```

### 입력 언어 변경

`listen()` 함수의 기본 language 파라미터 변경:

```python
def listen(timeout_seconds: int = 300, language: str = "en") -> str:
# "ko" = 한국어, "en" = 영어, "ja" = 일본어
```

### 전체 영어 설정 예시

```python
# get_tts()
_tts = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

# listen()
def listen(timeout_seconds: int = 300, language: str = "en") -> str:

# speak()
def speak(text: str, voice: str = "af_heart", speed: float = 1.0) -> str:
    """Speak in English. Text must be in English only!"""
```

## 프로젝트 구조

```
vtuber/
├── voice_mcp.py      # MCP 서버 메인
├── setup_models.py   # 모델 사전 다운로드
├── echo.py           # 독립 실행 버전 (Ollama 연동)
├── run.sh            # echo.py 실행 스크립트
├── test_vad.py       # VAD 테스트 도구
├── requirements.txt  # 의존성
└── README.md
```

## 동작 원리

### 전체 플로우

```
[사용자] --말함--> [마이크] --오디오--> [Silero VAD] --음성구간--> [Whisper] --텍스트--> [Claude]
                                                                                         |
[사용자] <--듣기-- [스피커] <--오디오-- [Kokoro TTS] <--텍스트------------------------------+
```

### 1. 음성 감지 (VAD)

```python
# Silero VAD가 음성 확률 계산 (0.0 ~ 1.0)
speech_prob = vad_model(chunk_tensor, SAMPLE_RATE)

# RMS(볼륨)도 함께 체크해서 배경 소음 필터링
rms = np.sqrt(np.mean(chunk ** 2))

# 둘 다 임계값 넘어야 음성으로 인식
is_voice = speech_prob > 0.85 and rms > 0.015
```

### 2. 음성 인식 (STT)

```python
# MLX Whisper - Apple Silicon GPU 가속
result = mlx_whisper.transcribe(
    audio_data,
    path_or_hf_repo="mlx-community/whisper-medium-mlx",
    language="ko"  # 한국어
)
```

### 3. 음성 합성 (TTS)

```python
# Kokoro TTS - 일본어 음성 생성
tts = KPipeline(lang_code='j', repo_id='hexgrad/Kokoro-82M')
for _, _, audio in tts(text, voice='jf_alpha', speed=1.0):
    sd.play(audio, 24000)
```

### MCP 통신

```
Claude Code <--stdio--> voice_mcp.py (FastMCP 서버)
                            |
                            ├── listen()   # 도구 1
                            ├── speak()    # 도구 2
                            └── listen_fixed()  # 도구 3
```

MCP (Model Context Protocol)는 Claude Code가 외부 도구를 호출할 수 있게 해주는 프로토콜입니다. `~/.mcp.json`에 서버를 등록하면 Claude가 해당 도구들을 사용할 수 있습니다.

## Claude Code에서 사용하기

### 1. MCP 서버 등록 확인

Claude Code 실행 후 `/mcp` 입력:

```
> /mcp
✓ voice (connected)
```

`voice`가 connected 상태면 준비 완료.

### 2. 음성 대화 시작

```
> listen
```

입력하면:
1. 비프음 (듣기 시작)
2. 말하기
3. 비프음 (인식 시작)
4. Claude가 텍스트로 응답 + 일본어 음성 재생
5. 대화 계속 또는 종료

### 3. 대화 예시

```
> listen

⏺ voice - listen (MCP)
  ⎿ { "result": "[사용자]: 안녕하세요\n\n⚠️ ..." }

⏺ voice - speak (MCP)(text: "こんにちは！何かお手伝いできますか？")
  ⎿ { "result": "→ listen() 호출하세요" }

⏺ voice - listen (MCP)
  ⎿ { "result": "[사용자]: 오늘 날씨 어때?\n\n⚠️ ..." }

...
```

### 4. 음성 대화 종료

- "끝", "바이바이", "고마워" 등을 말하면 Claude가 대화 종료
- 또는 Ctrl+C로 강제 종료
- 타임아웃 (5분간 말 없으면 자동 종료)

### 5. 팁

- **첫 실행 시** 모델 로딩으로 시간이 걸림 (TTS가 "初期化中" 안내)
- **말할 때** 비프음 후 0.5초 정도 기다렸다가 말하기
- **말 끝날 때** 1.5초 정도 조용히 있으면 인식 시작
- **Claude 응답 후** 자동으로 다시 듣기 모드 (수동으로 listen 입력 필요할 수도 있음)
