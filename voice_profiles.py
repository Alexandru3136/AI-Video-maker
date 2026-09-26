"""Persistent voice-reference profile manager.

Saved profiles live in ROOT/voices/<profile_name>/ with the reference audio file
and a metadata JSON. Users can save a new profile after uploading a reference,
and reuse it in future runs without re-uploading.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICES_DIR = ROOT / "voices"


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    audio_path: Path
    language: str


def _meta_path(profile_dir: Path) -> Path:
    return profile_dir / "profile.json"


def list_profiles() -> list[VoiceProfile]:
    """Return all saved voice profiles, sorted by name."""
    if not VOICES_DIR.exists():
        return []
    profiles = []
    for entry in sorted(VOICES_DIR.iterdir()):
        meta = _meta_path(entry)
        if entry.is_dir() and meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            audio = entry / data.get("audio_filename", "")
            if audio.is_file():
                profiles.append(VoiceProfile(name=data["name"], audio_path=audio, language=data.get("language", "")))
    return profiles


MAX_REFERENCE_BYTES = 30 * 1024 * 1024  # 30 MB
MIN_REFERENCE_BYTES = 5 * 1024  # 5 KB


def save_profile(name: str, audio_bytes: bytes, audio_filename: str, language: str) -> VoiceProfile:
    """Save a new voice profile from uploaded audio bytes."""
    slug = name.strip().replace(" ", "_").lower()
    if not slug:
        raise ValueError("Profile name cannot be empty.")
    if len(audio_bytes) < MIN_REFERENCE_BYTES:
        raise ValueError(f"Reference file too small ({len(audio_bytes)} bytes). Minimum is {MIN_REFERENCE_BYTES} bytes.")
    if len(audio_bytes) > MAX_REFERENCE_BYTES:
        raise ValueError(f"Reference file too large ({len(audio_bytes) // 1024 // 1024} MB). Maximum is {MAX_REFERENCE_BYTES // 1024 // 1024} MB.")
    profile_dir = VOICES_DIR / slug
    profile_dir.mkdir(parents=True, exist_ok=True)
    audio_path = profile_dir / audio_filename
    audio_path.write_bytes(audio_bytes)
    meta = {"name": name.strip(), "audio_filename": audio_filename, "language": language}
    _meta_path(profile_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return VoiceProfile(name=name.strip(), audio_path=audio_path, language=language)


def delete_profile(name: str) -> bool:
    """Delete a saved voice profile by name."""
    slug = name.strip().replace(" ", "_").lower()
    profile_dir = VOICES_DIR / slug
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        return True
    return False


def get_profile(name: str) -> VoiceProfile | None:
    """Look up a profile by name."""
    for profile in list_profiles():
        if profile.name == name:
            return profile
    return None
