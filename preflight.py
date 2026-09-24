"""Preflight environment checks. Never prints secret values, only presence/status."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    required: bool
    detail: str


def _check_env_key(var: str, label: str, required: bool = True) -> CheckResult:
    present = bool(os.getenv(var, "").strip())
    detail = "configurat" if present else f"lipseste din .env ({var})"
    return CheckResult(label, present, required, detail)


def _check_ffmpeg() -> CheckResult:
    path = shutil.which("ffmpeg")
    return CheckResult("FFmpeg", path is not None, True, path or "nu este pe PATH")


def _check_gpu() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("GPU CUDA (PyTorch)", False, False, "torch neinstalat (necesar doar pentru Chatterbox)")
    if torch.cuda.is_available():
        return CheckResult("GPU CUDA (PyTorch)", True, False, torch.cuda.get_device_name(0))
    return CheckResult("GPU CUDA (PyTorch)", False, False, "CUDA indisponibil (Chatterbox va esua)")


def _check_comfyui_workflow() -> CheckResult:
    path = os.getenv("COMFYUI_WORKFLOW_PATH", "").strip()
    if not path:
        return CheckResult("Workflow ComfyUI", False, False, "COMFYUI_WORKFLOW_PATH nesetat (ComfyUI va fi sarit, se foloseste Pexels)")
    if not Path(path).expanduser().is_file():
        return CheckResult("Workflow ComfyUI", False, False, f"fisierul nu exista: {path}")
    return CheckResult("Workflow ComfyUI", True, False, "gasit")


def _check_comfyui_server() -> CheckResult:
    base_url = os.getenv("COMFYUI_SERVER_URL", "http://127.0.0.1:8188").rstrip("/")
    try:
        response = requests.get(f"{base_url}/system_stats", timeout=5)
        response.raise_for_status()
        return CheckResult("Server ComfyUI", True, False, base_url)
    except requests.RequestException:
        return CheckResult("Server ComfyUI", False, False, f"inaccesibil la {base_url} (ComfyUI va fi sarit)")


def _check_tts_engine(engine: str, language_code: str, has_reference: bool) -> CheckResult:
    if engine == "chatterbox":
        if not has_reference:
            return CheckResult("Motor TTS (Chatterbox)", False, True, "lipseste fisierul de referinta vocala")
        try:
            import chatterbox.mtl_tts  # noqa: F401
        except ImportError:
            return CheckResult("Motor TTS (Chatterbox)", False, True, "chatterbox neinstalat")
        return CheckResult("Motor TTS (Chatterbox)", True, True, "instalat")
    if engine == "piper":
        suffix = language_code.upper()
        model = os.getenv(f"PIPER_MODEL_PATH_{suffix}", os.getenv("PIPER_MODEL_PATH", "")).strip()
        config = os.getenv(f"PIPER_CONFIG_PATH_{suffix}", os.getenv("PIPER_CONFIG_PATH", "")).strip()
        if not (model and config and Path(model).expanduser().is_file() and Path(config).expanduser().is_file()):
            return CheckResult("Motor TTS (Piper)", False, True, f"model/config lipsa (PIPER_MODEL_PATH_{suffix})")
        return CheckResult("Motor TTS (Piper)", True, True, "model gasit")
    return CheckResult("Motor TTS (Edge)", True, True, "online, fara configurare")


def run_preflight(engine: str, language_code: str, has_reference: bool, needs_comfyui: bool) -> list[CheckResult]:
    """Return all relevant checks for the current run configuration."""
    checks = [
        _check_env_key("GEMINI_API_KEY", "Cheie Gemini"),
        _check_env_key("PEXELS_API_KEY", "Cheie Pexels"),
        _check_ffmpeg(),
        _check_tts_engine(engine, language_code, has_reference),
    ]
    if engine == "chatterbox":
        checks.append(_check_gpu())
    if needs_comfyui:
        checks.append(_check_comfyui_workflow())
        checks.append(_check_comfyui_server())
    return checks


def blocking_failures(checks: list[CheckResult]) -> list[CheckResult]:
    return [check for check in checks if check.required and not check.ok]
