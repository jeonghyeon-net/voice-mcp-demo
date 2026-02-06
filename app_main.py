#!/usr/bin/env python3
"""Voice MCP Launcher - GUI 의존성 없는 macOS 런처."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from runtime_config import RuntimeConfig, load_runtime_config, save_output_language


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _embedded_server_binary() -> Path:
    if getattr(sys, "frozen", False):
        app_resources = Path(sys.executable).resolve().parent.parent / "Resources"
        return app_resources / "voice-mcp-server" / "voice-mcp-server"
    return _project_root() / "dist" / "voice-mcp-server" / "voice-mcp-server"


def _server_command_for_mcp() -> tuple[str, list[str]]:
    embedded = _embedded_server_binary()
    if embedded.exists():
        return str(embedded), []

    # 개발 환경 fallback
    script = _project_root() / "voice_mcp_server.py"
    return sys.executable, [str(script)]


def install_mcp_config(config: RuntimeConfig) -> str:
    mcp_path = Path.home() / ".mcp.json"
    if mcp_path.exists():
        try:
            payload = json.loads(mcp_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}
    else:
        payload = {}

    payload.setdefault("mcpServers", {})
    command, args = _server_command_for_mcp()
    payload["mcpServers"]["voice"] = {
        "command": command,
        "args": args,
    }
    mcp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(mcp_path)


def _run_osascript(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )


def _escape_osascript_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _show_alert(message: str, title: str = "Voice MCP Launcher") -> None:
    title_escaped = _escape_osascript_text(title)
    message_escaped = _escape_osascript_text(message)
    script = (
        f'display alert "{title_escaped}" message "{message_escaped}" as informational '
        'buttons {"확인"} default button "확인"'
    )
    _run_osascript(script)


def _select_language_with_dialog(default_language: str) -> str | None:
    default_label = "한국어(ko)" if default_language == "ko" else "일본어(ja)"
    script = (
        'button returned of (display dialog "출력 언어를 선택하세요." '
        'buttons {"취소", "일본어(ja)", "한국어(ko)"} '
        f'default button "{default_label}")'
    )
    proc = _run_osascript(script)
    if proc.returncode != 0:
        return None
    answer = proc.stdout.strip()
    if answer == "한국어(ko)":
        return "ko"
    if answer == "일본어(ja)":
        return "ja"
    return None


def _open_log(config: RuntimeConfig) -> None:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    if not config.log_file.exists():
        config.log_file.write_text("", encoding="utf-8")
    subprocess.run(["open", str(config.log_file)], check=False)


def run_launcher_wizard() -> None:
    config = load_runtime_config()
    selected = _select_language_with_dialog(default_language=config.output_language)
    if selected is None:
        return

    save_output_language(selected)
    updated = load_runtime_config()
    mcp_path = install_mcp_config(updated)
    _show_alert(
        "설정 완료\n"
        f"- 출력 언어: {selected}\n"
        f"- MCP 설정 파일: {mcp_path}\n\n"
        "Claude Code를 재시작한 뒤 /mcp에서 voice 연결을 확인하세요."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice MCP Launcher")
    parser.add_argument("--serve-mcp", action="store_true", help="MCP 서버 모드로 실행")
    parser.add_argument("--install-mcp", action="store_true", help="~/.mcp.json 갱신")
    parser.add_argument("--set-language", choices=["ko", "ja"], help="출력 언어 설정")
    parser.add_argument("--open-log", action="store_true", help="로그 파일 열기")
    parser.add_argument("--no-dialog", action="store_true", help="다이얼로그 없이 CLI로 실행")
    args = parser.parse_args()

    if args.serve_mcp:
        from voice_mcp import run_server

        run_server()
        return

    changed = False
    if args.set_language:
        save_output_language(args.set_language)
        changed = True

    config = load_runtime_config()

    if args.install_mcp:
        path = install_mcp_config(config)
        print(f"MCP 설정 갱신 완료: {path}")
        changed = True

    if args.open_log:
        _open_log(config)
        changed = True

    if changed:
        return

    if args.no_dialog:
        print("작업이 지정되지 않았습니다. --help를 확인하세요.")
        return

    run_launcher_wizard()


if __name__ == "__main__":
    main()
