"""Pluggable narration engines for local and fallback TTS generation."""
from __future__ import annotations

import asyncio
import os
import threading
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import edge_tts

from script_engine import NarrationBeat

TTSEngine = Literal["chatterbox", "piper", "edge"]

LANGUAGE_CODES = {
    "Romanian": "ro",
    "English": "en",
    "Russian": "ru",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}
CHATTERBOX_LANGUAGES = {"en", "ru", "fr", "es", "de"}
EDGE_VOICES = {
    "ro": "ro-RO-EmilNeural",
    "en": "en-US-ChristopherNeural",
    "ru": "ru-RU-DmitryNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "de": "de-DE-ConradNeural",
}


@dataclass(frozen=True)
class TTSConfig:
    engine: TTSEngine
    language: str
    reference_audio: Path | None = None
    chatterbox_variant: str = "v3"

    @property
    def language_code(self) -> str:
        try:
            return LANGUAGE_CODES[self.language]
        except KeyError as exc:
            raise ValueError(f"Unsupported narration language: {self.language}.") from exc


def available_engines(language: str) -> list[str]:
    code = LANGUAGE_CODES[language]
    engines = ["Piper (local)", "Edge (online fallback)"]
    if code in CHATTERBOX_LANGUAGES:
        engines.insert(0, "Chatterbox V3 (local)")
    return engines


def engine_from_label(label: str) -> TTSEngine:
    if label.startswith("Chatterbox"):
        return "chatterbox"
    if label.startswith("Piper"):
        return "piper"
    return "edge"


async def _save_edge_tts(text: str, voice: str, destination: Path) -> None:
    await edge_tts.Communicate(text=text, voice=voice).save(str(destination))


def _run_edge_batch(beats: list[NarrationBeat], language_code: str, output_dir: Path) -> list[Path]:
    voice = EDGE_VOICES[language_code]

    async def generate() -> list[Path]:
        paths = [output_dir / f"beat_{beat.beat_id:04d}.mp3" for beat in beats]
        await asyncio.gather(*[_save_edge_tts(beat.narration_text, voice, path) for beat, path in zip(beats, paths)])
        return paths

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(generate())

    paths: list[Path] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            paths.extend(asyncio.run(generate()))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return paths


@lru_cache(maxsize=2)
def _load_chatterbox(variant: str):
    try:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:
        raise RuntimeError("Chatterbox is not installed. Activate its environment and run: pip install -e .") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Chatterbox did not detect an NVIDIA CUDA GPU. Check the PyTorch CUDA installation.")
    return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=variant)


def _generate_chatterbox(beats: list[NarrationBeat], config: TTSConfig, output_dir: Path) -> list[Path]:
    if config.language_code not in CHATTERBOX_LANGUAGES:
        raise ValueError("Chatterbox V3 does not officially support Romanian. Select Piper for Romanian narration.")
    if not config.reference_audio or not config.reference_audio.is_file():
        raise ValueError("Chatterbox requires a valid 10-30 second voice-reference audio file.")
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("Chatterbox dependency torchaudio is missing. Reinstall Chatterbox with: pip install -e .") from exc

    model = _load_chatterbox(config.chatterbox_variant)
    paths: list[Path] = []
    for beat in beats:
        destination = output_dir / f"beat_{beat.beat_id:04d}.wav"
        audio = model.generate(
            beat.narration_text,
            language_id=config.language_code,
            audio_prompt_path=str(config.reference_audio),
            exaggeration=0.45,
            cfg_weight=0.35,
        )
        torchaudio.save(str(destination), audio.cpu(), model.sr)
        paths.append(destination)
    return paths


def _piper_model_paths(language_code: str) -> tuple[Path, Path]:
    suffix = language_code.upper()
    model_path = Path(os.getenv(f"PIPER_MODEL_PATH_{suffix}", os.getenv("PIPER_MODEL_PATH", ""))).expanduser()
    config_path = Path(os.getenv(f"PIPER_CONFIG_PATH_{suffix}", os.getenv("PIPER_CONFIG_PATH", ""))).expanduser()
    if not model_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"Piper model for {language_code} is not configured. Set PIPER_MODEL_PATH_{suffix} and PIPER_CONFIG_PATH_{suffix} in .env.")
    return model_path, config_path


@lru_cache(maxsize=8)
def _load_piper(model_path: str, config_path: str):
    try:
        from piper.voice import PiperVoice
    except ImportError as exc:
        raise RuntimeError("Piper is not installed. Run: pip install piper-tts") from exc
    return PiperVoice.load(model_path, config_path=config_path)


def _generate_piper(beats: list[NarrationBeat], config: TTSConfig, output_dir: Path) -> list[Path]:
    model_path, config_path = _piper_model_paths(config.language_code)
    voice = _load_piper(str(model_path), str(config_path))
    paths: list[Path] = []
    for beat in beats:
        destination = output_dir / f"beat_{beat.beat_id:04d}.wav"
        with wave.open(str(destination), "wb") as wav_file:
            voice.synthesize_wav(beat.narration_text, wav_file)
        paths.append(destination)
    return paths


def generate_audio(beats: list[NarrationBeat], config: TTSConfig, output_dir: Path) -> list[Path]:
    """Generate one audio file per visual narration beat using the selected engine."""
    if not beats:
        raise ValueError("Cannot generate narration for an empty script.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.engine == "chatterbox":
        return _generate_chatterbox(beats, config, output_dir)
    if config.engine == "piper":
        return _generate_piper(beats, config, output_dir)
    return _run_edge_batch(beats, config.language_code, output_dir)
