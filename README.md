# Voice MCP Launcher (macOS)

한국어 음성 대화를 기본값으로 제공하는 MCP 서버/런처 프로젝트입니다.

- 기본 출력: 한국어 TTS (로컬 모델, `sherpa-onnx`)
- 롤백 출력: 일본어 Kokoro TTS (로컬 ONNX, `kokoro_onnx`)
- STT: MLX Whisper (로컬 모델 번들)
- VAD: Silero VAD

핵심 목표는 **사용자가 별도 모델 다운로드/스크립트 실행 없이 앱만 설치해서 실행**하는 것입니다.

---

## 빠른 실행 명령어

### 빌드 + 앱 실행

```bash
cd /Users/me/Desktop/voice-mcp-demo
python3.11 -m venv venv
source venv/bin/activate
./scripts/build_macos_app.sh
open /Users/me/Desktop/voice-mcp-demo/dist/VoiceMCPLauncher.app
```

### 일본어 롤백

```bash
cd /Users/me/Desktop/voice-mcp-demo
source venv/bin/activate
./venv/bin/python app_main.py --set-language ja --install-mcp
```

### 한국어로 복귀

```bash
cd /Users/me/Desktop/voice-mcp-demo
source venv/bin/activate
./venv/bin/python app_main.py --set-language ko --install-mcp
```

---

## 1. 현재 동작 방식

앱은 두 구성으로 빌드됩니다.

1. `VoiceMCPLauncher.app`  
   - 더블클릭 실행용 GUI 런처
   - 출력 언어 설정(ko/ja)
   - `~/.mcp.json`의 `voice` 서버 항목 자동 등록/갱신

2. `voice-mcp-server` (런처 앱 내부 리소스)  
   - Claude Code가 stdio로 직접 호출하는 MCP 서버 바이너리
   - Whisper/TTS 모델을 앱 패키지에 포함

---

## 2. 최종 사용자 사용 방법

> 아래는 빌드된 앱을 받은 사용자 기준입니다.

1. `VoiceMCPLauncher.app` 실행
2. 다이얼로그에서 출력 언어 선택
   - `ko`: 한국어 출력(기본)
   - `ja`: 일본어 Kokoro 출력(롤백)
3. 완료 알림 확인
4. Claude Code 재시작 후 `/mcp`에서 `voice` 연결 확인

추가 설치/모델 다운로드/`setup_models.py` 실행은 필요 없습니다.

런처는 CLI 모드도 지원합니다.

```bash
# 출력 언어만 변경
./venv/bin/python app_main.py --set-language ko

# MCP 설정만 갱신
./venv/bin/python app_main.py --install-mcp
```

---

## 3. 개발자 빌드 방법

### 3.1 준비

- macOS (Apple Silicon)
- Python 3.11+
- 가상환경 권장

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3.2 앱 빌드

```bash
./scripts/build_macos_app.sh
```

스크립트가 수행하는 작업:

1. 의존성 설치
2. 모델 준비 (`setup_models.py`, 빌드 전용)
3. MCP 서버 바이너리 빌드
4. 런처 `.app` 빌드

산출물:

- `dist/VoiceMCPLauncher.app`

---

## 4. 런타임 설정

런타임 설정 파일:

- `~/Library/Application Support/VoiceMCP/runtime.json`

기본 예시:

```json
{
  "output_language": "ko",
  "whisper": {
    "model_path": "assets/models/whisper-medium-mlx",
    "fallback_repo": "mlx-community/whisper-medium-mlx"
  },
  "ko_tts": {
    "model_dir": "assets/models/vits-mimic3-ko_KO-kss_low",
    "speaker_id": 0,
    "speed": 1.0
  },
  "ja_tts": {
    "model_path": "assets/models/kokoro/kokoro-v1.0.onnx",
    "voices_path": "assets/models/kokoro/voices-v1.0.bin",
    "voice": "jf_alpha",
    "speed": 1.0
  }
}
```

---

## 5. 일본어 롤백 방법

1. `VoiceMCPLauncher.app` 실행
2. 출력 언어를 `ja`로 저장
3. Claude Code 재시작

이렇게 하면 `speak()`는 Kokoro 일본어 백엔드로 동작합니다.

---

## 6. 프로젝트 구조

```text
voice-mcp-demo/
├── app_main.py                 # 런처 앱 엔트리포인트
├── voice_mcp.py                # MCP 서버 본체
├── voice_mcp_server.py         # MCP 서버 실행 엔트리포인트
├── runtime_config.py           # 런타임 설정/경로 유틸
├── setup_models.py             # 빌드 전용 모델 준비
├── scripts/
│   └── build_macos_app.sh      # 앱 빌드 스크립트
├── assets/
│   └── models/                 # 번들 대상 모델(빌드 시 생성)
├── requirements.txt
└── README.md
```

---

## 7. 트러블슈팅

### 마이크가 동작하지 않는 경우

1. macOS 시스템 설정에서 마이크 권한 확인
2. Claude Code(또는 실행 주체 앱)의 마이크 권한 허용 확인
3. 로그 확인:  
   `~/Library/Application Support/VoiceMCP/voice_debug.log`

### MCP 연결이 안 되는 경우

1. 런처 앱을 다시 실행해 설정 마법사 재완료
2. `~/.mcp.json`의 `mcpServers.voice.command` 경로 확인
3. Claude Code 재시작 후 `/mcp` 재확인

---

## 8. 참고

- `echo.py`는 레거시 실험 스크립트입니다.
- 현재 권장 실행 경로는 `VoiceMCPLauncher.app` + 내장 `voice-mcp-server`입니다.
