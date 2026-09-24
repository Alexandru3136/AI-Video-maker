"""Unit tests for the Chatterbox narration path, which the UI file_uploader cannot exercise.

These cover the validation that runs *before* the model loads, so no GPU/torch is needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tts_engine import (
    TTSConfig,
    available_engines,
    engine_from_label,
    generate_audio,
)
from script_engine import NarrationBeat


def _beat() -> NarrationBeat:
    return NarrationBeat(
        beat_id=1,
        narration_text="A valid narration beat of sufficient length.",
        pexels_keywords="rome",
        fallback_ai_prompt="A cinematic Roman forum at dawn.",
    )


def test_available_engines_excludes_chatterbox_for_romanian():
    engines = available_engines("Romanian")
    assert "Chatterbox V3 (local)" not in engines
    assert engines == ["Piper (local)", "Edge (online fallback)"]


def test_available_engines_offers_chatterbox_first_for_english():
    engines = available_engines("English")
    assert engines[0] == "Chatterbox V3 (local)"


def test_engine_from_label_mapping():
    assert engine_from_label("Chatterbox V3 (local)") == "chatterbox"
    assert engine_from_label("Piper (local)") == "piper"
    assert engine_from_label("Edge (online fallback)") == "edge"


def test_language_code_rejects_unsupported_language():
    config = TTSConfig(engine="edge", language="Klingon")
    with pytest.raises(ValueError, match="Unsupported narration language"):
        _ = config.language_code


def test_chatterbox_rejects_romanian(tmp_path):
    config = TTSConfig(engine="chatterbox", language="Romanian", reference_audio=None)
    with pytest.raises(ValueError, match="does not officially support Romanian"):
        generate_audio([_beat()], config, tmp_path)


def test_chatterbox_requires_reference_audio(tmp_path):
    config = TTSConfig(engine="chatterbox", language="English", reference_audio=None)
    with pytest.raises(ValueError, match="voice-reference audio file"):
        generate_audio([_beat()], config, tmp_path)


def test_chatterbox_rejects_missing_reference_file(tmp_path):
    missing = tmp_path / "nope.wav"
    config = TTSConfig(engine="chatterbox", language="English", reference_audio=missing)
    with pytest.raises(ValueError, match="voice-reference audio file"):
        generate_audio([_beat()], config, tmp_path)


def test_generate_audio_rejects_empty_beat_list(tmp_path):
    config = TTSConfig(engine="edge", language="English")
    with pytest.raises(ValueError, match="empty script"):
        generate_audio([], config, tmp_path)
