"""Tests for the persistent voice-profile manager."""
from __future__ import annotations

import voice_profiles


def test_save_list_get_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")

    # Empty at start.
    assert voice_profiles.list_profiles() == []

    # Save a profile.
    profile = voice_profiles.save_profile("Test Voice", b"\x00" * 8192, "ref.wav", "English")
    assert profile.name == "Test Voice"
    assert profile.audio_path.is_file()

    # List it.
    profiles = voice_profiles.list_profiles()
    assert len(profiles) == 1
    assert profiles[0].name == "Test Voice"
    assert profiles[0].language == "English"

    # Get by name.
    found = voice_profiles.get_profile("Test Voice")
    assert found is not None and found.name == "Test Voice"
    assert voice_profiles.get_profile("Nonexistent") is None

    # Delete it.
    assert voice_profiles.delete_profile("Test Voice") is True
    assert voice_profiles.list_profiles() == []
    assert voice_profiles.delete_profile("Test Voice") is False


def test_save_rejects_empty_name(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
    import pytest
    with pytest.raises(ValueError, match="empty"):
        voice_profiles.save_profile("   ", b"\x00", "ref.wav", "English")


def test_save_rejects_too_small_file(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
    import pytest
    with pytest.raises(ValueError, match="too small"):
        voice_profiles.save_profile("Tiny", b"\x00" * 100, "ref.wav", "English")


def test_multiple_profiles_sorted(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_profiles, "VOICES_DIR", tmp_path / "voices")
    voice_profiles.save_profile("Beta Voice", b"\x00" * 8192, "b.wav", "French")
    voice_profiles.save_profile("Alpha Voice", b"\x00" * 8192, "a.wav", "English")
    profiles = voice_profiles.list_profiles()
    assert len(profiles) == 2
    # Sorted by name (directory name, which is lowercase slug).
    assert profiles[0].name == "Alpha Voice"
    assert profiles[1].name == "Beta Voice"
