"""Comprehensive audit tests covering edge cases, bugs, and untested paths.

Organized by module. Each test targets a specific gap found during code review.
"""
from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, seconds: float, rate: int = 22050) -> Path:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))
    return path


def _beat(beat_id=1, text="A valid narration beat of sufficient length.", key=False):
    from script_engine import NarrationBeat
    return NarrationBeat(
        beat_id=beat_id,
        narration_text=text,
        pexels_keywords="ancient rome colosseum",
        fallback_ai_prompt="A cinematic wide shot of ancient Roman ruins at dawn.",
        is_key_action_moment=key,
    )


# ===========================================================================
# script_engine
# ===========================================================================

class TestSimplifiedSchema:
    """The _simplified_schema() must be Gemini-compatible (no min/max constraints)."""

    def test_schema_has_no_constraints(self):
        from script_engine import _simplified_schema
        schema_str = json.dumps(_simplified_schema())
        for banned in ["minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"]:
            assert banned not in schema_str, f"Schema contains {banned} — Gemini will reject it"

    def test_schema_has_required_fields(self):
        from script_engine import _simplified_schema
        schema = _simplified_schema()
        assert "title" in schema["properties"]
        assert "scenes" in schema["properties"]
        beat_props = schema["properties"]["scenes"]["items"]["properties"]["beats"]["items"]["properties"]
        assert "narration_text" in beat_props
        assert "pexels_keywords" in beat_props
        assert "fallback_ai_prompt" in beat_props
        assert "is_key_action_moment" in beat_props

    def test_generate_script_still_validates_with_pydantic(self, monkeypatch):
        """Even with simplified schema, Pydantic validation must still run."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        import script_engine
        bad_json = json.dumps({"title": "X", "scenes": []})
        response = mock.Mock(text=bad_json)
        client = mock.Mock()
        client.models.generate_content.return_value = response
        monkeypatch.setattr(script_engine.genai, "Client", lambda api_key: client)
        with pytest.raises(RuntimeError, match="Gemini failed"):
            script_engine.generate_script("Roman trade routes", "English", 12)

    def test_strip_json_handles_markdown_fence(self):
        from script_engine import _strip_json
        raw = '```json\n{"title": "Test"}\n```'
        assert json.loads(_strip_json(raw)) == {"title": "Test"}

    def test_strip_json_rejects_non_json(self):
        from script_engine import _strip_json
        with pytest.raises(ValueError, match="JSON"):
            _strip_json("no json here at all")


# ===========================================================================
# voice_profiles edge cases
# ===========================================================================

class TestVoiceProfilesEdgeCases:
    def test_save_rejects_too_large_file(self, tmp_path, monkeypatch):
        import voice_profiles
        monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
        large = b"\x00" * (31 * 1024 * 1024)
        with pytest.raises(ValueError, match="too large"):
            voice_profiles.save_profile("Big", large, "ref.wav", "English")

    def test_save_overwrites_existing_profile(self, tmp_path, monkeypatch):
        import voice_profiles
        monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
        voice_profiles.save_profile("Same", b"\x00" * 8192, "a.wav", "English")
        voice_profiles.save_profile("Same", b"\xff" * 8192, "b.wav", "French")
        profiles = voice_profiles.list_profiles()
        assert len(profiles) == 1
        assert profiles[0].language == "French"

    def test_delete_nonexistent_returns_false(self, tmp_path, monkeypatch):
        import voice_profiles
        monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
        assert voice_profiles.delete_profile("ghost") is False

    def test_get_profile_returns_none_for_missing(self, tmp_path, monkeypatch):
        import voice_profiles
        monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
        assert voice_profiles.get_profile("nope") is None

    def test_list_profiles_ignores_corrupt_dir(self, tmp_path, monkeypatch):
        import voice_profiles
        monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
        corrupt = tmp_path / "voices" / "broken_profile"
        corrupt.mkdir(parents=True)
        (corrupt / "random_file.txt").write_text("not a profile")
        assert voice_profiles.list_profiles() == []

    def test_save_rejects_path_traversal(self, tmp_path, monkeypatch):
        import voice_profiles
        monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
        with pytest.raises(ValueError, match="invalid characters"):
            voice_profiles.save_profile("../../etc", b"\x00" * 8192, "ref.wav", "English")

    def test_slug_handles_special_characters(self, tmp_path, monkeypatch):
        import voice_profiles
        monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
        p = voice_profiles.save_profile("My Voice!", b"\x00" * 8192, "ref.wav", "English")
        assert p.name == "My Voice!"
        assert voice_profiles.get_profile("My Voice!") is not None


# ===========================================================================
# media_engine
# ===========================================================================

class TestMediaEngine:
    def test_plan_visual_sources_single_beat(self):
        from media_engine import VisualSource, plan_visual_sources
        beats = [_beat(1, key=True)]
        plan = plan_visual_sources(beats)
        assert len(plan) == 1

    def test_plan_visual_sources_all_key_beats(self):
        from media_engine import VisualSource, plan_visual_sources
        beats = [_beat(i, key=True) for i in range(1, 11)]
        plan = plan_visual_sources(beats)
        comfy_count = sum(1 for v in plan.values() if v == VisualSource.COMFYUI)
        assert comfy_count == round(10 * 0.40)

    def test_acquire_visual_falls_back_to_pollinations(self, tmp_path, monkeypatch):
        import media_engine
        monkeypatch.setattr(media_engine, "generate_comfyui_video", lambda *a: False)
        monkeypatch.setattr(media_engine, "download_pexels_video", lambda *a, **k: (False, None))
        monkeypatch.setattr(media_engine, "download_pollinations_image", lambda prompt, dest, w, h: dest.write_bytes(b"\x00" * 2048) or True)
        from media_engine import VisualSource, acquire_visual
        path, kind, source = acquire_visual(_beat(), VisualSource.COMFYUI, tmp_path, "16:9 Long-Form", 5.0)
        assert kind == "image"
        assert source == VisualSource.IMAGE.value

    def test_acquire_visual_raises_when_all_fail(self, tmp_path, monkeypatch):
        import media_engine
        monkeypatch.setattr(media_engine, "generate_comfyui_video", lambda *a: False)
        monkeypatch.setattr(media_engine, "download_pexels_video", lambda *a, **k: (False, None))
        monkeypatch.setattr(media_engine, "download_pollinations_image", lambda *a, **k: False)
        from media_engine import VisualSource, acquire_visual
        with pytest.raises(RuntimeError, match="Could not acquire"):
            acquire_visual(_beat(), VisualSource.PEXELS, tmp_path, "16:9 Long-Form", 5.0)

    def test_build_scene_media_mismatched_lengths(self, tmp_path):
        from media_engine import build_scene_media
        with pytest.raises(ValueError, match="exactly one"):
            build_scene_media([_beat()], [], tmp_path, "16:9 Long-Form")

    def test_download_returns_false_for_small_file(self, tmp_path, monkeypatch):
        import media_engine
        def fake_get(url, stream=True, timeout=90):
            resp = mock.Mock()
            resp.raise_for_status = lambda: None
            resp.raw = mock.Mock()
            resp.raw.read = lambda size=None: b"\x00" * 100
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            import io
            resp.raw = io.BytesIO(b"\x00" * 100)
            return resp
        monkeypatch.setattr(media_engine.requests, "get", fake_get)
        result = media_engine._download("http://example.com/file", tmp_path / "out.bin")
        assert result is False

    def test_pollinations_retries_on_failure(self, tmp_path, monkeypatch):
        import media_engine
        calls = {"n": 0}
        def fake_download(url, dest, timeout=90):
            calls["n"] += 1
            return False
        monkeypatch.setattr(media_engine, "_download", fake_download)
        result = media_engine.download_pollinations_image("test", tmp_path / "img.jpg")
        assert result is False
        assert calls["n"] == 2


# ===========================================================================
# video_composer
# ===========================================================================

class TestVideoComposer:
    def test_compose_empty_raises(self):
        from video_composer import compose_video
        with pytest.raises(ValueError, match="empty"):
            compose_video([], Path("out.mp4"), "16:9 Long-Form")

    def test_shorts_uses_portrait_dimensions(self, tmp_path):
        """Verify 9:16 Shorts produces 1080x1920 dimensions."""
        from video_composer import compose_video
        from media_engine import SceneMedia
        audio = _write_wav(tmp_path / "a.wav", 1.0)
        img = tmp_path / "v.jpg"
        from PIL import Image
        Image.new("RGB", (1920, 1080), color="red").save(str(img))
        media = [SceneMedia(_beat(), audio, img, "image", "pexels", "pexels")]
        out = tmp_path / "out.mp4"
        compose_video(media, out, "9:16 Shorts")
        assert out.exists() and out.stat().st_size > 0
        from moviepy import VideoFileClip
        clip = VideoFileClip(str(out))
        try:
            assert clip.w == 1080
            assert clip.h == 1920
        finally:
            clip.close()

    def test_landscape_uses_correct_dimensions(self, tmp_path):
        from video_composer import compose_video
        from media_engine import SceneMedia
        audio = _write_wav(tmp_path / "a.wav", 1.0)
        img = tmp_path / "v.jpg"
        from PIL import Image
        Image.new("RGB", (1920, 1080), color="blue").save(str(img))
        media = [SceneMedia(_beat(), audio, img, "image", "pexels", "pexels")]
        out = tmp_path / "out.mp4"
        compose_video(media, out, "16:9 Long-Form")
        assert out.exists()
        from moviepy import VideoFileClip
        clip = VideoFileClip(str(out))
        try:
            assert clip.w == 1920
            assert clip.h == 1080
        finally:
            clip.close()


# ===========================================================================
# project_artifacts
# ===========================================================================

class TestProjectArtifacts:
    def test_srt_timestamp_format(self):
        from project_artifacts import _srt_timestamp
        assert _srt_timestamp(0) == "00:00:00,000"
        assert _srt_timestamp(61.5) == "00:01:01,500"
        assert _srt_timestamp(3661.123) == "01:01:01,123"

    def test_srt_timestamp_negative_clamps_to_zero(self):
        from project_artifacts import _srt_timestamp
        assert _srt_timestamp(-5) == "00:00:00,000"

    def test_timeline_manifest_roundtrip(self, tmp_path):
        from project_artifacts import write_timeline_manifest
        from media_engine import SceneMedia
        audio = _write_wav(tmp_path / "a.wav", 1.0)
        visual = tmp_path / "v.jpg"
        visual.write_bytes(b"\xff" * 100)
        media = [SceneMedia(_beat(42, text="Specific narration line here."), audio, visual, "video", "comfyui", "pexels")]
        dest = write_timeline_manifest(media, tmp_path / "tl.json")
        data = json.loads(dest.read_text(encoding="utf-8"))
        assert data["beats"][0]["beat_id"] == 42
        assert data["beats"][0]["narration_text"] == "Specific narration line here."
        assert data["beats"][0]["visual_kind"] == "video"


# ===========================================================================
# tts_engine
# ===========================================================================

class TestTTSEngine:
    def test_engine_from_label_unknown_defaults_to_edge(self):
        from tts_engine import engine_from_label
        assert engine_from_label("Some Unknown Engine") == "edge"

    def test_available_engines_all_languages(self):
        from tts_engine import available_engines, LANGUAGE_CODES
        for lang in LANGUAGE_CODES:
            engines = available_engines(lang)
            assert len(engines) >= 2
            assert "Edge (online fallback)" in engines

    def test_tts_config_language_code_all_valid(self):
        from tts_engine import TTSConfig, LANGUAGE_CODES
        for lang, code in LANGUAGE_CODES.items():
            config = TTSConfig(engine="edge", language=lang)
            assert config.language_code == code


# ===========================================================================
# preflight
# ===========================================================================

class TestPreflight:
    def test_blocking_failures_filters_correctly(self):
        from preflight import CheckResult, blocking_failures
        checks = [
            CheckResult("OK Required", True, True, "fine"),
            CheckResult("OK Optional", True, False, "fine"),
            CheckResult("FAIL Required", False, True, "broken"),
            CheckResult("FAIL Optional", False, False, "meh"),
        ]
        blockers = blocking_failures(checks)
        assert len(blockers) == 1
        assert blockers[0].name == "FAIL Required"

    def test_comfyui_workflow_relative_path_resolves(self, tmp_path, monkeypatch):
        from preflight import _check_comfyui_workflow
        workflow = tmp_path / "test_workflow.json"
        workflow.write_text("{}")
        monkeypatch.setenv("COMFYUI_WORKFLOW_PATH", str(workflow))
        result = _check_comfyui_workflow()
        assert result.ok is True

    def test_comfyui_workflow_missing_env(self, monkeypatch):
        from preflight import _check_comfyui_workflow
        monkeypatch.delenv("COMFYUI_WORKFLOW_PATH", raising=False)
        result = _check_comfyui_workflow()
        assert result.ok is False
        assert "nesetat" in result.detail

    def test_run_preflight_includes_gpu_for_chatterbox(self, monkeypatch):
        from preflight import run_preflight
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setenv("PEXELS_API_KEY", "k")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
        checks = run_preflight("chatterbox", "en", True, False)
        names = [c.name for c in checks]
        assert "GPU CUDA (PyTorch)" in names

    def test_run_preflight_excludes_gpu_for_edge(self, monkeypatch):
        from preflight import run_preflight
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setenv("PEXELS_API_KEY", "k")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
        checks = run_preflight("edge", "en", False, False)
        names = [c.name for c in checks]
        assert "GPU CUDA (PyTorch)" not in names


# ===========================================================================
# youtube_upload
# ===========================================================================

class TestYouTubeUpload:
    def test_is_available_false_without_lib(self, monkeypatch):
        import youtube_upload
        monkeypatch.delenv("YOUTUBE_CLIENT_SECRETS", raising=False)
        assert youtube_upload.is_available() is False

    def test_is_available_false_with_missing_file(self, monkeypatch):
        import youtube_upload
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS", "/nonexistent/path/secrets.json")
        with mock.patch.dict("sys.modules", {"google_auth_oauthlib": mock.Mock(), "googleapiclient": mock.Mock()}):
            assert youtube_upload.is_available() is False


# ===========================================================================
# comfyui integration (media_engine)
# ===========================================================================

class TestComfyUI:
    def test_load_workflow_raises_without_env(self, monkeypatch):
        import media_engine
        monkeypatch.delenv("COMFYUI_WORKFLOW_PATH", raising=False)
        with pytest.raises(RuntimeError, match="not configured"):
            media_engine._load_workflow()

    def test_inject_comfy_prompt_uses_configured_node(self, monkeypatch):
        import media_engine
        monkeypatch.setenv("COMFYUI_POSITIVE_PROMPT_NODE_ID", "7")
        workflow = {"7": {"inputs": {"text": "old"}}}
        media_engine._inject_comfy_prompt(workflow, "new prompt")
        assert workflow["7"]["inputs"]["text"] == "new prompt"

    def test_inject_comfy_prompt_raises_for_bad_node_id(self, monkeypatch):
        import media_engine
        monkeypatch.setenv("COMFYUI_POSITIVE_PROMPT_NODE_ID", "999")
        workflow = {"7": {"inputs": {"text": "old"}}}
        with pytest.raises(RuntimeError, match="999"):
            media_engine._inject_comfy_prompt(workflow, "test")

    def test_inject_comfy_prompt_finds_clip_text_encode(self, monkeypatch):
        import media_engine
        monkeypatch.setenv("COMFYUI_POSITIVE_PROMPT_NODE_ID", "")
        workflow = {
            "3": {"class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}, "inputs": {"text": "old"}},
            "4": {"class_type": "CLIPTextEncode", "_meta": {"title": "negative"}, "inputs": {"text": "neg"}},
        }
        media_engine._inject_comfy_prompt(workflow, "my prompt")
        assert workflow["3"]["inputs"]["text"] == "my prompt"
        assert workflow["4"]["inputs"]["text"] == "neg"

    def test_inject_comfy_prompt_no_clip_node_raises(self, monkeypatch):
        import media_engine
        monkeypatch.setenv("COMFYUI_POSITIVE_PROMPT_NODE_ID", "")
        workflow = {"1": {"class_type": "KSampler", "inputs": {}}}
        with pytest.raises(RuntimeError, match="CLIPTextEncode"):
            media_engine._inject_comfy_prompt(workflow, "test")

    def test_generate_comfyui_video_handles_connection_error(self, monkeypatch, tmp_path):
        import media_engine
        monkeypatch.setenv("COMFYUI_WORKFLOW_PATH", str(tmp_path / "wf.json"))
        (tmp_path / "wf.json").write_text(json.dumps({"3": {"class_type": "CLIPTextEncode", "_meta": {"title": "positive"}, "inputs": {"text": ""}}}))
        monkeypatch.setenv("COMFYUI_POSITIVE_PROMPT_NODE_ID", "")
        import requests as req
        monkeypatch.setattr(req, "post", mock.Mock(side_effect=req.ConnectionError("refused")))
        result = media_engine.generate_comfyui_video("test prompt", tmp_path / "out.mp4")
        assert result is False


# ===========================================================================
# app.py UI edge cases (via AppTest)
# ===========================================================================

class TestAppUIEdgeCases:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("PEXELS_API_KEY", "test-key")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
        import preflight
        from preflight import CheckResult
        monkeypatch.setattr(
            preflight, "_check_comfyui_server",
            lambda: CheckResult("Server ComfyUI", False, False, "stub"),
        )

    def test_title_is_header_not_title(self):
        """After UI fix, main page should use st.header not st.title."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60).run()
        assert not at.exception
        has_header = any("Autonomous AI Video Generator" in h.value for h in at.header)
        has_title = any("Autonomous AI Video Generator" in t.value for t in at.title)
        assert has_header, "Main page should use st.header"
        assert not has_title, "Main page should NOT use st.title for the generator heading"

    def test_romanian_is_first_language(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60).run()
        assert at.selectbox[0].options[0] == "Romanian"
