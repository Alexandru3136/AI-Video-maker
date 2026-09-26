"""Gemini-backed, validated documentary script generation."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from google import genai
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


class Scene(BaseModel):
    scene_id: int = Field(ge=1)
    narration_text: str = Field(min_length=20, description="Full scene narration, assembled from its beats.")
    beats: list["NarrationBeat"] = Field(min_length=6, max_length=16)


class NarrationBeat(BaseModel):
    beat_id: int = Field(default=0, ge=0)
    narration_text: str = Field(min_length=12, max_length=280)
    pexels_keywords: str = Field(min_length=2)
    fallback_ai_prompt: str = Field(min_length=10)
    is_key_action_moment: bool = False


Scene.model_rebuild()


class VideoScript(BaseModel):
    title: str = Field(min_length=3)
    scenes: list[Scene] = Field(min_length=4)


SYSTEM_PROMPT = """You are an elite YouTube documentary writer and visual editor. Return only valid JSON that exactly
matches the supplied schema. Create an engaging, accurate long-form documentary script in the requested
language. Use a cold open, escalating narrative, clear transitions, concrete details, and a satisfying
ending. Every scene contains 6-16 narration beats. A beat is one short, continuous spoken thought, normally
3-6 seconds (roughly 10-22 words). Its narration_text is the exact narration spoken during that visual.
The scene narration_text must be the natural concatenation of its beats. Never reuse a visual beat just to
fill time: each beat needs a distinct visual concept that genuinely supports its narration. Do not use markdown,
citations, stage directions, or labels in narration_text. pexels_keywords must be simple English search keywords.
fallback_ai_prompt must be a vivid, safe English cinematic visual prompt. Mark is_key_action_moment true only
for visually exceptional action moments; target approximately 40% of all beats, but never mark adjacent beats.
The remaining beats will use Pexels or images."""


def _simplified_schema() -> dict:
    """Return a Gemini-compatible JSON schema without constraints that cause too-many-states errors."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scene_id": {"type": "integer"},
                        "narration_text": {"type": "string"},
                        "beats": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "beat_id": {"type": "integer"},
                                    "narration_text": {"type": "string"},
                                    "pexels_keywords": {"type": "string"},
                                    "fallback_ai_prompt": {"type": "string"},
                                    "is_key_action_moment": {"type": "boolean"},
                                },
                                "required": ["narration_text", "pexels_keywords", "fallback_ai_prompt", "is_key_action_moment"],
                            },
                        },
                    },
                    "required": ["scene_id", "narration_text", "beats"],
                },
            },
        },
        "required": ["title", "scenes"],
    }


def _strip_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last < first:
        raise ValueError("Gemini did not return a JSON object.")
    return text[first:last + 1]


def generate_script(topic: str, language: str, target_minutes: int) -> VideoScript:
    """Generate and validate the scene plan with Gemini's free-tier-capable Flash model."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env before generating a script.")
    if not topic or len(topic.strip()) < 4:
        raise ValueError("Enter a more specific video topic.")

    scene_count = max(8, target_minutes)
    prompt = (
        f"Topic: {topic.strip()}\nNarration language: {language}\nTarget runtime: {target_minutes} minutes.\n"
        f"Create exactly {scene_count} scenes. The total narration should closely match the target runtime."
    )
    client = genai.Client(api_key=key)
    max_attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_json_schema": _simplified_schema(),
                    "temperature": 0.75,
                },
            )
            script = VideoScript.model_validate(json.loads(_strip_json(response.text)))
            break
        except (ValidationError, json.JSONDecodeError, ValueError, Exception) as exc:
            last_error = exc
            log.warning("Gemini attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Gemini failed after {max_attempts} attempts: {last_error}") from last_error

    next_beat_id = 1
    for index, scene in enumerate(script.scenes, start=1):
        scene.scene_id = index
        scene.narration_text = " ".join(beat.narration_text.strip() for beat in scene.beats)
        for beat in scene.beats:
            beat.beat_id = next_beat_id
            next_beat_id += 1

    beats = [beat for scene in script.scenes for beat in scene.beats]
    key_limit = max(1, round(len(beats) * 0.40))
    marked_indexes = [index for index, beat in enumerate(beats) if beat.is_key_action_moment]
    # Avoid adjacent ComfyUI requests even when the model marks them consecutively.
    allowed_indexes: set[int] = set()
    for index in marked_indexes + list(range(0, len(beats), 2)) + list(range(1, len(beats), 2)):
        if len(allowed_indexes) >= key_limit:
            break
        if index - 1 not in allowed_indexes and index + 1 not in allowed_indexes:
            allowed_indexes.add(index)
    for index, beat in enumerate(beats):
        beat.is_key_action_moment = index in allowed_indexes
    return script
