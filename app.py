from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from media_engine import build_scene_media
from preflight import blocking_failures, run_preflight
from project_artifacts import write_subtitles, write_timeline_manifest
from script_engine import generate_script
from tts_engine import LANGUAGE_CODES, TTSConfig, available_engines, engine_from_label, generate_audio
from video_composer import compose_video

load_dotenv()
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="AI Video Generator", page_icon="video_camera", layout="centered")
st.title("Autonomous AI Video Generator")
st.caption("Gemini + local Chatterbox/Piper TTS + Pexels/Pollinations + optional ComfyUI.")
st.caption("Timeline vizual: 40% ComfyUI, 35% Pexels, 25% imagini AI. Fara loop video artificial.")

# These inputs intentionally remain outside a Streamlit form. The engine list must refresh
# immediately when the user changes the narration language.
topic = st.text_area("Tema videoclipului", placeholder="Ex.: Cum a schimbat Imperiul Roman comertul in Europa", height=120)
language = st.selectbox("Limba nararii", ["Romanian", "English", "Russian", "French", "Spanish", "German"])
aspect_ratio = st.radio("Format video", ["16:9 Long-Form", "9:16 Shorts"], horizontal=True)
smoke_test = st.checkbox("Smoke test rapid (~1 min, primele cateva beats)", value=False, help="Valideaza intreg pipeline-ul rapid si ieftin inainte de un video complet.")
duration = 1 if smoke_test else st.slider("Durata tinta (minute)", min_value=10, max_value=20, value=12)
tts_label = st.selectbox("Motor TTS", available_engines(language))
reference_audio = None
if engine_from_label(tts_label) == "chatterbox":
    reference_audio = st.file_uploader("Inregistrare de referinta (WAV/MP3/FLAC, 10-30 sec)", type=["wav", "mp3", "flac"])

engine = engine_from_label(tts_label)
has_reference = reference_audio is not None
with st.expander("Verificari preflight", expanded=False):
    st.caption("Include un test de retea catre ComfyUI (pana la 5s). Ruleaza la cerere.")
    if st.button("Ruleaza verificarile"):
        checks = run_preflight(engine, LANGUAGE_CODES[language], has_reference, needs_comfyui=True)
        for check in checks:
            icon = "OK" if check.ok else ("EROARE" if check.required else "atentie")
            line = f"**{check.name}** — {check.detail}"
            if check.ok:
                st.success(f"[{icon}] {line}")
            elif check.required:
                st.error(f"[{icon}] {line}")
            else:
                st.warning(f"[{icon}] {line}")

submitted = st.button("Genereaza Videoclip", type="primary", use_container_width=True)

if submitted:
    if not topic.strip():
        st.error("Introdu tema videoclipului.")
        st.stop()

    blockers = blocking_failures(run_preflight(engine, LANGUAGE_CODES[language], has_reference, needs_comfyui=False))
    if blockers:
        st.error("Nu pot porni. Rezolva mai intai: " + "; ".join(f"{b.name} ({b.detail})" for b in blockers))
        st.stop()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS / run_id
    progress = st.progress(0, text="Pregatire...")
    status = st.empty()
    try:
        status.write("1/4 - Generez scenariul structurat cu Gemini...")
        script = generate_script(topic, language, duration)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "script.json").write_text(json.dumps(script.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        progress.progress(20, text="Scenariu generat")

        status.write("2/4 - Generez naratiunea audio...")
        reference_path = None
        if reference_audio is not None:
            reference_path = run_dir / "reference" / reference_audio.name
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_bytes(reference_audio.getvalue())
        tts_config = TTSConfig(engine=engine_from_label(tts_label), language=language, reference_audio=reference_path)
        beats = [beat for scene in script.scenes for beat in scene.beats]
        if smoke_test:
            # Keep the full script.json on disk, but only render the first few beats end-to-end.
            beats = beats[:4]
        audio_paths = generate_audio(beats, tts_config, run_dir / "audio")
        write_subtitles(beats, audio_paths, run_dir / "subtitles.srt")
        progress.progress(40, text="Audio generat")

        status.write("3/4 - Colectez vizualurile...")

        def update_visual_progress(index: int, total: int, planned: str, actual: str) -> None:
            completed = 40 + int(index / total * 30)
            progress.progress(completed, text=f"Vizual {index}/{total}: {actual} (planificat: {planned})")

        media = build_scene_media(beats, audio_paths, run_dir / "visuals", aspect_ratio, update_visual_progress)
        write_timeline_manifest(media, run_dir / "timeline.json")
        progress.progress(70, text="Vizualuri pregatite")

        status.write("4/4 - Randare finala locala. Aceasta poate dura cateva minute...")
        output = compose_video(media, run_dir / "final_video.mp4", aspect_ratio)
        progress.progress(100, text="Videoclip finalizat")
        status.success(f"Gata: {output.name}")
        st.video(str(output))
        with output.open("rb") as video_file:
            st.download_button("Descarca MP4", video_file, file_name=output.name, mime="video/mp4")
        with st.expander("Scenariu generat"):
            st.json(script.model_dump())
    except Exception as exc:
        progress.empty()
        status.error(f"Generarea s-a oprit: {exc}")
        if run_dir.exists() and not any(run_dir.iterdir()):
            shutil.rmtree(run_dir)
