"""Audio-first MoviePy composition with no artificial video looping and Ken Burns motion."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from moviepy import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips

if TYPE_CHECKING:
    from media_engine import SceneMedia


FPS = 24


def _fit_frame(clip, width: int, height: int):
    scale = max(width / clip.w, height / clip.h)
    resized = clip.resized(scale)
    return resized.cropped(x_center=resized.w / 2, y_center=resized.h / 2, width=width, height=height)


def _ken_burns(image_path: Path, duration: float, width: int, height: int):
    clip = ImageClip(str(image_path)).with_duration(duration)
    # Overscan creates room for a subtle zoom while preserving a full frame crop.
    base = _fit_frame(clip, int(width * 1.10), int(height * 1.10))
    def frame_function(get_frame, t):
        factor = 1.0 + 0.04 * min(t / max(duration, 0.01), 1.0)
        frame = get_frame(t)
        from PIL import Image
        image = Image.fromarray(frame)
        scaled = image.resize((int(image.width * factor), int(image.height * factor)), Image.Resampling.LANCZOS)
        left = max(0, (scaled.width - width) // 2)
        top = max(0, (scaled.height - height) // 2)
        return np.asarray(scaled.crop((left, top, left + width, top + height)))
    return base.transform(frame_function).with_duration(duration)


def _trim_video(path: Path, duration: float, width: int, height: int):
    source = _fit_frame(VideoFileClip(str(path)).without_audio(), width, height)
    if source.duration < duration:
        source.close()
        raise RuntimeError(f"Video asset is shorter than its narration beat: {path}")
    return source.subclipped(0, duration), source


def compose_video(scene_media: list["SceneMedia"], output_path: Path, aspect_ratio: str, progress_callback=None) -> Path:
    if not scene_media:
        raise ValueError("Cannot compose an empty video.")
    width, height = (1080, 1920) if aspect_ratio == "9:16 Shorts" else (1920, 1080)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_clips, resources = [], []
    try:
        for index, item in enumerate(scene_media, start=1):
            audio = AudioFileClip(str(item.audio_path))
            duration = audio.duration
            resources.append(audio)
            if item.visual_kind == "image":
                visual = _ken_burns(item.visual_path, duration, width, height)
            else:
                visual, source = _trim_video(item.visual_path, duration, width, height)
                resources.append(source)
            clip = visual.with_audio(audio).with_duration(duration)
            resources.extend([visual, clip])
            scene_clips.append(clip)
            if progress_callback:
                progress_callback(index, len(scene_media))
        final = concatenate_videoclips(scene_clips, method="compose")
        resources.append(final)
        final.write_videofile(str(output_path), fps=FPS, codec="libx264", audio_codec="aac", preset="medium", threads=4, logger=None)
        return output_path
    finally:
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception:
                pass
