"""Portable output artifacts for a completed generation run."""
from __future__ import annotations

import json
from pathlib import Path

from moviepy import AudioFileClip

from media_engine import SceneMedia
from script_engine import NarrationBeat


def _srt_timestamp(seconds: float) -> str:
    total_ms = round(max(0, seconds) * 1000)
    hours, remaining = divmod(total_ms, 3_600_000)
    minutes, remaining = divmod(remaining, 60_000)
    secs, milliseconds = divmod(remaining, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def write_subtitles(beats: list[NarrationBeat], audio_paths: list[Path], destination: Path) -> Path:
    if len(beats) != len(audio_paths):
        raise ValueError("Each subtitle beat needs an audio file.")
    cursor = 0.0
    entries: list[str] = []
    for index, (beat, audio_path) in enumerate(zip(beats, audio_paths), start=1):
        audio = AudioFileClip(str(audio_path))
        try:
            end = cursor + audio.duration
        finally:
            audio.close()
        entries.append(f"{index}\n{_srt_timestamp(cursor)} --> {_srt_timestamp(end)}\n{beat.narration_text.strip()}\n")
        cursor = end
    destination.write_text("\n".join(entries), encoding="utf-8")
    return destination


def write_timeline_manifest(media: list[SceneMedia], destination: Path) -> Path:
    payload = {
        "beats": [
            {
                "beat_id": item.beat.beat_id,
                "narration_text": item.beat.narration_text,
                "planned_source": item.planned_source,
                "actual_source": item.actual_source,
                "audio_path": str(item.audio_path),
                "visual_path": str(item.visual_path),
                "visual_kind": item.visual_kind,
            }
            for item in media
        ]
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
