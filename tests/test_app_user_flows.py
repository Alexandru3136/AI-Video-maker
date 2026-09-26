"""End-to-end user-interaction tests driving the real Streamlit app via AppTest.

Each test simulates a real user: typing a topic, choosing language/format/engine,
and clicking buttons. Heavy external services (Gemini, TTS, media download, MoviePy
render) are replaced with faithful fakes that still create real files, so the whole
app.py control flow runs exactly as it would for a user, minus the network/GPU cost.
"""
from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(ROOT / "app.py")
OUTPUTS = ROOT / "outputs"


# --- Faithful fakes for the backend engines -------------------------------------

def _fake_script():
    from script_engine import NarrationBeat, Scene, VideoScript

    scenes = []
    beat_id = 1
    for scene_id in range(1, 5):
        beats = []
        for _ in range(6):
            beats.append(
                NarrationBeat(
                    beat_id=beat_id,
                    narration_text="This is a valid narration beat of sufficient length.",
                    pexels_keywords="ancient rome",
                    fallback_ai_prompt="A cinematic wide shot of a Roman forum at dawn.",
                    is_key_action_moment=(beat_id % 3 == 0),
                )
            )
            beat_id += 1
        scenes.append(
            Scene(
                scene_id=scene_id,
                narration_text=" ".join(b.narration_text for b in beats),
                beats=beats,
            )
        )
    return VideoScript(title="Test Documentary", scenes=scenes)


def _fake_generate_script(topic, language, target_minutes):
    return _fake_script()


def _fake_generate_audio(beats, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for beat in beats:
        path = output_dir / f"beat_{beat.beat_id:04d}.mp3"
        path.write_bytes(b"\x00" * 2048)
        paths.append(path)
    return paths


def _fake_build_scene_media(beats, audio_paths, output_dir, aspect_ratio, progress_callback=None):
    from media_engine import SceneMedia

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    media = []
    for index, (beat, audio_path) in enumerate(zip(beats, audio_paths), start=1):
        visual = output_dir / f"beat_{beat.beat_id:04d}.jpg"
        visual.write_bytes(b"\x00" * 2048)
        media.append(SceneMedia(beat, Path(audio_path), visual, "image", "pexels", "pexels"))
        if progress_callback:
            progress_callback(index, len(beats), "pexels", "pexels")
    return media


def _fake_compose_video(media, output_path, aspect_ratio, progress_callback=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"\x00" * 4096)
    return output_path


@contextlib.contextmanager
def patched_backend(record=None):
    """Patch every heavy engine at its source module so app.py's fresh imports pick fakes."""
    gen_audio = mock.Mock(side_effect=_fake_generate_audio)
    build_media = mock.Mock(side_effect=_fake_build_scene_media)
    if record is not None:
        record["generate_audio"] = gen_audio
        record["build_scene_media"] = build_media
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("script_engine.generate_script", _fake_generate_script))
        stack.enter_context(mock.patch("tts_engine.generate_audio", gen_audio))
        stack.enter_context(mock.patch("media_engine.build_scene_media", build_media))
        stack.enter_context(mock.patch("video_composer.compose_video", _fake_compose_video))
        stack.enter_context(mock.patch("project_artifacts.write_subtitles", lambda *a, **k: Path(a[2])))
        stack.enter_context(mock.patch("project_artifacts.write_timeline_manifest", lambda *a, **k: Path(a[1])))
        stack.enter_context(mock.patch("media_engine.audio_duration", return_value=5.0))
        yield


@pytest.fixture(autouse=True)
def clean_outputs():
    """Remove only the run directories created during a test."""
    before = set(OUTPUTS.iterdir()) if OUTPUTS.exists() else set()
    yield
    if OUTPUTS.exists():
        for entry in OUTPUTS.iterdir():
            if entry not in before and entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)


@pytest.fixture(autouse=True)
def fast_env(monkeypatch):
    """Give the app a healthy environment: keys present, ffmpeg found, no ComfyUI network wait."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    import preflight
    from preflight import CheckResult
    monkeypatch.setattr(
        preflight,
        "_check_comfyui_server",
        lambda: CheckResult("Server ComfyUI", False, False, "stub: skipped in tests"),
    )


def _new_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    return at


def _generate_button(at: AppTest):
    for button in at.button:
        if button.label == "Genereaza Videoclip":
            return button
    raise AssertionError("Generate button not found")


def _radio(at: AppTest, label: str):
    for r in at.radio:
        if r.label == label:
            return r
    raise AssertionError(f"Radio '{label}' not found")


# --- Tests ----------------------------------------------------------------------

def test_app_loads_without_error():
    """A user opens the app: it renders title and inputs, no exceptions."""
    at = _new_app().run()
    assert not at.exception
    assert any("Autonomous AI Video Generator" in t.value for t in at.title)
    assert _generate_button(at).label == "Genereaza Videoclip"


def test_engine_list_reacts_to_language():
    """Romanian excludes Chatterbox; switching to English adds it."""
    at = _new_app().run()
    assert at.selectbox[0].value == "Romanian"
    ro_engines = at.selectbox[1].options
    assert "Chatterbox V3 (local)" not in ro_engines
    assert "Piper (local)" in ro_engines and "Edge (online fallback)" in ro_engines

    at.selectbox[0].set_value("English").run()
    en_engines = at.selectbox[1].options
    assert "Chatterbox V3 (local)" in en_engines


def test_preflight_runs_only_on_demand():
    """Preflight checks appear only after clicking the button, not on every rerun."""
    at = _new_app().run()
    # No check result rendered before the button is pressed.
    assert not any("Cheie Gemini" in s.value for s in at.success)
    check_button = next(b for b in at.button if b.label == "Ruleaza verificarile")
    check_button.click().run()
    assert not at.exception
    assert any("Cheie Gemini" in s.value for s in at.success)


def test_empty_topic_shows_error():
    """User clicks generate with no topic: friendly error, no backend work."""
    with patched_backend() as _:
        at = _new_app().run()
        _generate_button(at).click().run()
    assert not at.exception
    assert any("Introdu tema videoclipului" in e.value for e in at.error)


def test_preflight_blocks_when_gemini_key_missing(monkeypatch):
    """With the Gemini key removed, generation is blocked before any engine runs."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    record = {}
    with patched_backend(record):
        at = _new_app().run()
        at.text_area[0].set_value("Roman Empire trade routes").run()
        _generate_button(at).click().run()
    assert not at.exception
    assert any("Nu pot porni" in e.value for e in at.error)
    record["generate_audio"].assert_not_called()


def test_happy_path_generates_video():
    """Full real-user flow: type topic, keep Romanian + Edge, generate, get an MP4."""
    with patched_backend() as _:
        at = _new_app().run()
        at.text_area[0].set_value("How Rome reshaped European trade").run()
        at.selectbox[1].set_value("Edge (online fallback)").run()
        _radio(at, "Format video").set_value("16:9 Long-Form").run()
        _generate_button(at).click().run()
    assert not at.exception
    assert any("Gata:" in s.value for s in at.success)


def test_smoke_test_limits_beats_to_four():
    """Enabling the smoke-test checkbox renders only the first 4 beats end-to-end."""
    record = {}
    with patched_backend(record):
        at = _new_app().run()
        at.text_area[0].set_value("Quick pipeline validation").run()
        at.checkbox[0].set_value(True).run()
        at.selectbox[1].set_value("Edge (online fallback)").run()
        _generate_button(at).click().run()
    assert not at.exception
    beats_arg = record["generate_audio"].call_args.args[0]
    assert len(beats_arg) == 4


def test_shorts_format_flows_through_to_media():
    """Choosing 9:16 Shorts passes that aspect ratio to the media builder."""
    record = {}
    with patched_backend(record):
        at = _new_app().run()
        at.text_area[0].set_value("Vertical short about Rome").run()
        _radio(at, "Format video").set_value("9:16 Shorts").run()
        at.selectbox[1].set_value("Edge (online fallback)").run()
        _generate_button(at).click().run()
    assert not at.exception
    aspect_arg = record["build_scene_media"].call_args.args[3]
    assert aspect_arg == "9:16 Shorts"


def test_engine_failure_shows_friendly_error_not_crash():
    """If an engine raises mid-run, the app surfaces a friendly message and does not crash."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("script_engine.generate_script", _fake_generate_script))
        stack.enter_context(
            mock.patch("tts_engine.generate_audio", side_effect=RuntimeError("TTS backend exploded"))
        )
        at = _new_app().run()
        at.text_area[0].set_value("A topic that will fail during narration").run()
        at.selectbox[1].set_value("Edge (online fallback)").run()
        _generate_button(at).click().run()
    assert not at.exception
    assert any("Generarea s-a oprit" in e.value for e in at.error)


def test_history_page_shows_empty_state(tmp_path):
    """Navigating to Istoric with no prior runs shows an info message."""
    # Temporarily rename outputs/ so the history page sees an empty state.
    backup = None
    if OUTPUTS.exists():
        backup = OUTPUTS.with_name("outputs_backup_test")
        OUTPUTS.rename(backup)
    try:
        at = _new_app().run()
        _radio(at, "Navigare").set_value("Istoric").run()
        assert not at.exception
        assert any("Nicio generare anterioara" in i.value for i in at.info)
    finally:
        if backup and backup.exists():
            if OUTPUTS.exists():
                shutil.rmtree(OUTPUTS)
            backup.rename(OUTPUTS)


def test_history_page_shows_past_run():
    """After generating a video, the history page lists it."""
    with patched_backend() as _:
        at = _new_app().run()
        at.text_area[0].set_value("History test topic").run()
        at.selectbox[1].set_value("Edge (online fallback)").run()
        _generate_button(at).click().run()
    assert not at.exception
    # Switch to history page — the run should be listed.
    _radio(at, "Navigare").set_value("Istoric").run()
    assert not at.exception
    assert at.info == []  # no empty-state message


def test_voice_profiles_page_loads():
    """The voice profiles page renders without error."""
    at = _new_app().run()
    _radio(at, "Navigare").set_value("Profiluri vocale").run()
    assert not at.exception
    assert any("Profiluri vocale" in t.value for t in at.title)
