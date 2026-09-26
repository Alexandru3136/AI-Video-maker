"""Hybrid visual acquisition for each distinct narration beat."""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import quote_plus

import requests
from moviepy import AudioFileClip, VideoFileClip

from script_engine import NarrationBeat

log = logging.getLogger(__name__)

@dataclass
class SceneMedia:
    beat: NarrationBeat
    audio_path: Path
    visual_path: Path
    visual_kind: str  # video or image
    planned_source: str
    actual_source: str


class VisualSource(str, Enum):
    COMFYUI = "comfyui"
    PEXELS = "pexels"
    IMAGE = "image"


def plan_visual_sources(beats: list[NarrationBeat]) -> dict[int, VisualSource]:
    """Build a deterministic 40/35/25 visual plan before downloading any asset."""
    if not beats:
        return {}

    beat_count = len(beats)
    comfy_target = round(beat_count * 0.40)
    image_target = round(beat_count * 0.25)
    key_beats = [beat for beat in beats if beat.is_key_action_moment]
    comfy_ids = {beat.beat_id for beat in key_beats[:comfy_target]}
    plan = {beat.beat_id: VisualSource.COMFYUI if beat.beat_id in comfy_ids else VisualSource.PEXELS for beat in beats}

    candidates = [beat for beat in beats if beat.beat_id not in comfy_ids]
    # Evenly spread static image moments across the timeline instead of grouping them.
    if candidates and image_target:
        image_target = min(image_target, len(candidates))
        selected_indexes = {min(len(candidates) - 1, round((index + 0.5) * len(candidates) / image_target - 0.5)) for index in range(image_target)}
        for index in selected_indexes:
            plan[candidates[index].beat_id] = VisualSource.IMAGE
    return plan


def _download(url: str, destination: Path, timeout: int = 90) -> bool:
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)
        return destination.exists() and destination.stat().st_size > 1024
    except requests.RequestException:
        destination.unlink(missing_ok=True)
        return False


def download_pexels_video(
    keywords: str,
    destination: Path,
    minimum_duration: float = 0,
    portrait: bool = False,
    used_video_ids: set[int] | None = None,
) -> tuple[bool, int | None]:
    """Download a Pexels video. Returns (success, video_id) for de-duplication."""
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return False, None
    orientation = "portrait" if portrait else "landscape"
    target_width, target_height = (1080, 1920) if portrait else (1920, 1080)
    if used_video_ids is None:
        used_video_ids = set()
    try:
        response = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": key},
            params={"query": keywords, "per_page": 15, "orientation": orientation},
            timeout=20,
        )
        response.raise_for_status()
        videos = response.json().get("videos", [])
        candidates = []
        for video in videos:
            video_id = int(video.get("id", 0))
            if video_id in used_video_ids:
                continue
            duration = float(video.get("duration", 0))
            if duration < minimum_duration:
                continue
            for file in video.get("video_files", []):
                width, height = file.get("width", 0), file.get("height", 0)
                if file.get("file_type") != "video/mp4":
                    continue
                if portrait and height >= 1280 and height > width:
                    candidates.append((file, duration, video_id))
                elif not portrait and width >= 1280 and width >= height:
                    candidates.append((file, duration, video_id))
        if not candidates:
            return False, None
        chosen, _, chosen_id = min(candidates, key=lambda item: abs(item[0].get("width", target_width) - target_width) + abs(item[0].get("height", target_height) - target_height))
        if _download(chosen["link"], destination):
            return True, chosen_id
        return False, None
    except (requests.RequestException, KeyError, ValueError):
        return False, None


def download_pollinations_image(prompt: str, destination: Path, width: int = 1920, height: int = 1080) -> bool:
    for attempt in range(2):
        seed = random.randint(1, 2_147_483_647)
        url = f"https://image.pollinations.ai/prompt/{quote_plus(prompt)}?width={width}&height={height}&seed={seed}&model=flux&nologo=true"
        if _download(url, destination, timeout=120):
            return True
        log.warning("Pollinations attempt %d failed for prompt: %.60s", attempt + 1, prompt)
    return False


def _load_workflow() -> dict:
    path = os.getenv("COMFYUI_WORKFLOW_PATH", "").strip()
    if not path:
        raise RuntimeError("COMFYUI_WORKFLOW_PATH is not configured.")
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _inject_comfy_prompt(workflow: dict, prompt: str) -> None:
    configured_node = os.getenv("COMFYUI_POSITIVE_PROMPT_NODE_ID", "").strip()
    if configured_node:
        try:
            workflow[configured_node]["inputs"]["text"] = prompt
            return
        except KeyError as exc:
            raise RuntimeError(f"COMFYUI_POSITIVE_PROMPT_NODE_ID={configured_node} was not found in the workflow.") from exc

    # Standard Wan workflows use CLIPTextEncode for prompts. Prefer a node whose title explicitly says positive.
    prompt_nodes = [(node_id, node) for node_id, node in workflow.items() if node.get("class_type") == "CLIPTextEncode"]
    if not prompt_nodes:
        raise RuntimeError("The ComfyUI API workflow contains no CLIPTextEncode node.")
    positive_node = next((node for _, node in prompt_nodes if "positive" in node.get("_meta", {}).get("title", "").lower()), prompt_nodes[0][1])
    positive_node.setdefault("inputs", {})["text"] = prompt


def generate_comfyui_video(prompt: str, destination: Path) -> bool:
    """Queue an exported API workflow, replace its first positive CLIP prompt, and download its first video."""
    base_url = os.getenv("COMFYUI_SERVER_URL", "http://127.0.0.1:8188").rstrip("/")
    try:
        workflow = _load_workflow()
        _inject_comfy_prompt(workflow, prompt)
        client_id = str(uuid.uuid4())
        queued = requests.post(f"{base_url}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
        queued.raise_for_status()
        prompt_id = queued.json()["prompt_id"]
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            history = requests.get(f"{base_url}/history/{prompt_id}", timeout=20).json().get(prompt_id, {})
            outputs = history.get("outputs", {})
            for node_output in outputs.values():
                for item in node_output.get("gifs", []) + node_output.get("videos", []):
                    params = {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
                    return _download(f"{base_url}/view?" + requests.compat.urlencode(params), destination, timeout=180)
            time.sleep(3)
    except (OSError, KeyError, ValueError, requests.RequestException, RuntimeError):
        destination.unlink(missing_ok=True)
        return False
    return False


def audio_duration(audio_path: Path) -> float:
    audio = AudioFileClip(str(audio_path))
    try:
        return audio.duration
    finally:
        audio.close()


def _video_is_long_enough(video_path: Path, minimum_duration: float) -> bool:
    try:
        clip = VideoFileClip(str(video_path))
        try:
            return clip.duration >= minimum_duration
        finally:
            clip.close()
    except OSError:
        return False


def acquire_visual(
    beat: NarrationBeat,
    source: VisualSource,
    output_dir: Path,
    aspect_ratio: str,
    audio_duration: float,
    used_video_ids: set[int] | None = None,
) -> tuple[Path, str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"beat_{beat.beat_id:04d}"
    if used_video_ids is None:
        used_video_ids = set()

    if source == VisualSource.COMFYUI and generate_comfyui_video(beat.fallback_ai_prompt, stem.with_suffix(".mp4")) and _video_is_long_enough(stem.with_suffix(".mp4"), audio_duration):
        return stem.with_suffix(".mp4"), "video", VisualSource.COMFYUI.value
    stem.with_suffix(".mp4").unlink(missing_ok=True)

    portrait = aspect_ratio == "9:16 Shorts"
    if source in {VisualSource.COMFYUI, VisualSource.PEXELS}:
        ok, vid_id = download_pexels_video(beat.pexels_keywords, stem.with_suffix(".mp4"), audio_duration, portrait, used_video_ids)
        if ok:
            if vid_id is not None:
                used_video_ids.add(vid_id)
            return stem.with_suffix(".mp4"), "video", VisualSource.PEXELS.value

    width, height = (1080, 1920) if portrait else (1920, 1080)
    image_path = stem.with_suffix(".jpg")
    if not download_pollinations_image(beat.fallback_ai_prompt, image_path, width, height):
        raise RuntimeError(f"Could not acquire any visual for beat {beat.beat_id}.")
    return image_path, "image", VisualSource.IMAGE.value


def build_scene_media(beats: list[NarrationBeat], audio_paths: list[Path], output_dir: Path, aspect_ratio: str, progress_callback=None) -> list[SceneMedia]:
    if len(beats) != len(audio_paths):
        raise ValueError("Each visual beat must have exactly one narration file.")
    plan = plan_visual_sources(beats)
    media: list[SceneMedia] = []
    used_video_ids: set[int] = set()
    failed_beats: list[int] = []
    max_retries = 2

    for index, (beat, audio_path) in enumerate(zip(beats, audio_paths), start=1):
        source = plan[beat.beat_id]
        acquired = False
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                visual_path, visual_kind, actual_source = acquire_visual(
                    beat, source, output_dir, aspect_ratio, audio_duration(audio_path), used_video_ids,
                )
                media.append(SceneMedia(beat, audio_path, visual_path, visual_kind, source.value, actual_source))
                acquired = True
                break
            except Exception as exc:
                last_error = exc
                log.warning("Beat %d attempt %d failed: %s", beat.beat_id, attempt + 1, exc)
                # On retry, fall back to image source (cheapest/most reliable).
                source = VisualSource.IMAGE

        if not acquired:
            failed_beats.append(beat.beat_id)
            log.error("Beat %d failed permanently after %d retries: %s", beat.beat_id, max_retries, last_error)

        if progress_callback:
            status = actual_source if acquired else f"FAILED ({last_error})"
            progress_callback(index, len(beats), plan[beat.beat_id].value, status)

    if failed_beats:
        raise RuntimeError(f"Could not acquire visuals for beats: {failed_beats}. {len(media)}/{len(beats)} succeeded.")
    return media
