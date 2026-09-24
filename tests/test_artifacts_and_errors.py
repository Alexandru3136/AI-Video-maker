"""Integration tests for the on-disk artifacts (SRT + timeline.json) using real audio.

These exercise the real MoviePy/ffmpeg path that the UI tests deliberately stub out.
"""
from __future__ import annotations

import json
import re
import struct
import wave
from pathlib import Path

import pytest

from project_artifacts import write_subtitles, write_timeline_manifest
from media_engine import SceneMedia
from script_engine import NarrationBeat

SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}$")


def _write_wav(path: Path, seconds: float, rate: int = 22050) -> Path:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))
    return path


def _beat(beat_id: int, text: str) -> NarrationBeat:
    return NarrationBeat(
        beat_id=beat_id,
        narration_text=text,
        pexels_keywords="rome",
        fallback_ai_prompt="A cinematic Roman forum at dawn.",
    )


def test_write_subtitles_generates_sequential_srt(tmp_path):
    a1 = _write_wav(tmp_path / "a1.wav", 1.0)
    a2 = _write_wav(tmp_path / "a2.wav", 0.5)
    beats = [_beat(1, "First narration line."), _beat(2, "Second narration line.")]
    dest = write_subtitles(beats, [a1, a2], tmp_path / "subtitles.srt")

    content = dest.read_text(encoding="utf-8")
    blocks = [b for b in content.strip().split("\n\n") if b.strip()]
    assert len(blocks) == 2

    first_lines = blocks[0].splitlines()
    assert first_lines[0] == "1"
    start, arrow, end = first_lines[1].split(" ")
    assert arrow == "-->"
    assert SRT_TIME.match(start) and SRT_TIME.match(end)
    assert start == "00:00:00,000"
    assert "First narration line." in blocks[0]

    # The second subtitle must start exactly where the first ended (no gaps/overlap).
    second_start = blocks[1].splitlines()[1].split(" ")[0]
    assert second_start == end
    assert blocks[1].splitlines()[0] == "2"


def test_write_subtitles_rejects_length_mismatch(tmp_path):
    a1 = _write_wav(tmp_path / "a1.wav", 1.0)
    beats = [_beat(1, "First line here."), _beat(2, "Second line here.")]
    with pytest.raises(ValueError, match="each subtitle beat needs an audio file|audio file"):
        write_subtitles(beats, [a1], tmp_path / "out.srt")


def test_write_timeline_manifest_shape(tmp_path):
    audio = _write_wav(tmp_path / "a.wav", 1.0)
    visual = tmp_path / "v.jpg"
    visual.write_bytes(b"\x00" * 10)
    media = [SceneMedia(_beat(1, "Narration line here."), audio, visual, "image", "comfyui", "pexels")]
    dest = write_timeline_manifest(media, tmp_path / "timeline.json")

    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert list(payload.keys()) == ["beats"]
    entry = payload["beats"][0]
    assert entry["beat_id"] == 1
    assert entry["planned_source"] == "comfyui"
    assert entry["actual_source"] == "pexels"
    assert entry["visual_kind"] == "image"
    assert entry["narration_text"] == "Narration line here."
