"""Logic-level tests for visual planning, Gemini script post-processing, and downloads.

None of these touch the real network or GPU: the Gemini client and `requests` are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

import media_engine
import script_engine
from media_engine import VisualSource, plan_visual_sources
from script_engine import NarrationBeat, generate_script


# --- Visual planning (pure) -----------------------------------------------------

def _beats(count: int, key_every: int | None = None) -> list[NarrationBeat]:
    beats = []
    for i in range(1, count + 1):
        beats.append(
            NarrationBeat(
                beat_id=i,
                narration_text="A valid narration beat of sufficient length.",
                pexels_keywords="rome",
                fallback_ai_prompt="A cinematic Roman forum at dawn.",
                is_key_action_moment=bool(key_every and i % key_every == 0),
            )
        )
    return beats


def test_plan_visual_sources_empty():
    assert plan_visual_sources([]) == {}


def test_plan_visual_sources_respects_40_25_split():
    beats = _beats(20, key_every=2)  # 10 key beats available
    plan = plan_visual_sources(beats)
    counts = {source: list(plan.values()).count(source) for source in VisualSource}
    # 40% ComfyUI (from key beats), 25% images, remainder Pexels.
    assert counts[VisualSource.COMFYUI] == round(20 * 0.40)
    assert counts[VisualSource.IMAGE] == round(20 * 0.25)
    assert counts[VisualSource.PEXELS] == 20 - counts[VisualSource.COMFYUI] - counts[VisualSource.IMAGE]
    assert set(plan.keys()) == {b.beat_id for b in beats}


def test_plan_visual_sources_comfy_only_from_key_beats():
    beats = _beats(10, key_every=5)  # only beats 5 and 10 are key
    plan = plan_visual_sources(beats)
    comfy_ids = [bid for bid, src in plan.items() if src == VisualSource.COMFYUI]
    assert set(comfy_ids).issubset({5, 10})


def test_plan_visual_sources_no_key_beats_uses_pexels_and_images():
    beats = _beats(8)  # no key moments
    plan = plan_visual_sources(beats)
    assert VisualSource.COMFYUI not in plan.values()
    assert VisualSource.IMAGE in plan.values()
    assert VisualSource.PEXELS in plan.values()


# --- Gemini script post-processing (client mocked) ------------------------------

def _script_json(scene_count: int = 4, beats_per_scene: int = 6) -> str:
    scenes = []
    for s in range(scene_count):
        beats = [
            {
                "beat_id": 0,
                "narration_text": f"Narration beat number {s}-{b} with enough length.",
                "pexels_keywords": "ancient rome",
                "fallback_ai_prompt": "A cinematic Roman forum at dawn, wide angle.",
                # Mark many adjacent beats on purpose to test the anti-adjacency guard.
                "is_key_action_moment": True,
            }
            for b in range(beats_per_scene)
        ]
        scenes.append({"scene_id": s + 1, "narration_text": "placeholder narration text here", "beats": beats})
    return json.dumps({"title": "Mocked Documentary", "scenes": scenes})


@pytest.fixture
def mocked_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = mock.Mock()
    response.text = _script_json()
    client = mock.Mock()
    client.models.generate_content.return_value = response
    monkeypatch.setattr(script_engine.genai, "Client", lambda api_key: client)
    return client


def test_generate_script_renumbers_beats_and_scenes(mocked_gemini):
    script = generate_script("Roman trade", "English", 12)
    assert [scene.scene_id for scene in script.scenes] == [1, 2, 3, 4]
    all_beats = [beat for scene in script.scenes for beat in scene.beats]
    assert [beat.beat_id for beat in all_beats] == list(range(1, len(all_beats) + 1))


def test_generate_script_concatenates_scene_narration(mocked_gemini):
    script = generate_script("Roman trade", "English", 12)
    scene = script.scenes[0]
    assert scene.narration_text == " ".join(b.narration_text.strip() for b in scene.beats)


def test_generate_script_caps_and_deadjacents_key_moments(mocked_gemini):
    script = generate_script("Roman trade", "English", 12)
    all_beats = [beat for scene in script.scenes for beat in scene.beats]
    key_indexes = [i for i, beat in enumerate(all_beats) if beat.is_key_action_moment]
    # At most ~40% marked, and never two adjacent.
    assert len(key_indexes) <= max(1, round(len(all_beats) * 0.40))
    assert all(b - a > 1 for a, b in zip(key_indexes, key_indexes[1:]))


def test_generate_script_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        generate_script("Roman trade", "English", 12)


def test_generate_script_rejects_short_topic(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="more specific video topic"):
        generate_script("ab", "English", 12)


# --- Download helpers (requests mocked) -----------------------------------------

def test_pexels_returns_false_without_key(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    ok, vid_id = media_engine.download_pexels_video("rome", Path("x.mp4"))
    assert ok is False


def test_pexels_requests_portrait_orientation_for_shorts(monkeypatch, tmp_path):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["params"] = params
        resp = mock.Mock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: {"videos": []}  # empty -> returns False, but params are captured
        return resp

    monkeypatch.setattr(media_engine.requests, "get", fake_get)
    ok, _ = media_engine.download_pexels_video("rome", tmp_path / "v.mp4", 0, portrait=True)
    assert ok is False
    assert captured["params"]["orientation"] == "portrait"


def test_pexels_landscape_default(monkeypatch, tmp_path):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["params"] = params
        resp = mock.Mock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: {"videos": []}
        return resp

    monkeypatch.setattr(media_engine.requests, "get", fake_get)
    ok, _ = media_engine.download_pexels_video("rome", tmp_path / "v.mp4")
    assert captured["params"]["orientation"] == "landscape"


def test_pexels_picks_portrait_candidate(monkeypatch, tmp_path):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    videos = {
        "videos": [
            {
                "id": 42,
                "duration": 20,
                "video_files": [
                    {"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "http://land"},
                    {"file_type": "video/mp4", "width": 1080, "height": 1920, "link": "http://port"},
                ],
            }
        ]
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        resp = mock.Mock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: videos
        return resp

    downloaded = {}

    def fake_download(link, destination, timeout=90):
        downloaded["link"] = link
        return True

    monkeypatch.setattr(media_engine.requests, "get", fake_get)
    monkeypatch.setattr(media_engine, "_download", fake_download)
    ok, vid_id = media_engine.download_pexels_video("rome", tmp_path / "v.mp4", 0, portrait=True)
    assert ok is True
    assert vid_id == 42
    assert downloaded["link"] == "http://port"


def test_pollinations_url_encodes_prompt(monkeypatch, tmp_path):
    captured = {}

    def fake_download(url, destination, timeout=90):
        captured["url"] = url
        return True

    monkeypatch.setattr(media_engine, "_download", fake_download)
    media_engine.download_pollinations_image("a roman forum", tmp_path / "i.jpg", 1080, 1920)
    assert "image.pollinations.ai" in captured["url"]
    assert "width=1080" in captured["url"] and "height=1920" in captured["url"]
    assert "a+roman+forum" in captured["url"] or "a%20roman%20forum" in captured["url"]


def test_pexels_deduplication_skips_used_video(monkeypatch, tmp_path):
    """A video already in used_video_ids should be skipped."""
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    videos = {
        "videos": [
            {"id": 100, "duration": 20, "video_files": [{"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "http://v100"}]},
            {"id": 200, "duration": 20, "video_files": [{"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "http://v200"}]},
        ]
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        resp = mock.Mock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: videos
        return resp

    downloaded = {}
    def fake_download(link, destination, timeout=90):
        downloaded["link"] = link
        return True

    monkeypatch.setattr(media_engine.requests, "get", fake_get)
    monkeypatch.setattr(media_engine, "_download", fake_download)
    # Video 100 already used — should pick 200 instead.
    ok, vid_id = media_engine.download_pexels_video("rome", tmp_path / "v.mp4", 0, False, used_video_ids={100})
    assert ok is True
    assert vid_id == 200
    assert downloaded["link"] == "http://v200"


def test_gemini_retry_succeeds_on_second_attempt(monkeypatch):
    """Gemini retry logic should recover from a transient failure."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    call_count = {"n": 0}

    def fake_generate(model, contents, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("transient network error")
        return mock.Mock(text=_script_json())

    client = mock.Mock()
    client.models.generate_content = fake_generate
    monkeypatch.setattr(script_engine.genai, "Client", lambda api_key: client)
    script = generate_script("Roman trade", "English", 12)
    assert call_count["n"] == 2
    assert len(script.scenes) == 4
