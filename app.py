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
from voice_profiles import VoiceProfile, delete_profile, list_profiles, save_profile

load_dotenv()
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="AI Video Generator", page_icon="video_camera", layout="wide")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigare", ["Genereaza", "Istoric", "Profiluri vocale"], label_visibility="collapsed")

# ===========================================================================
# PAGE: Voice profiles
# ===========================================================================
if page == "Profiluri vocale":
    st.title("Profiluri vocale")
    st.caption("Salveaza inregistrari de referinta pentru Chatterbox ca sa nu le reincarci de fiecare data.")

    profiles = list_profiles()
    if profiles:
        st.subheader("Profiluri salvate")
        for profile in profiles:
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.write(f"**{profile.name}** — {profile.language}")
            col2.audio(str(profile.audio_path))
            if col3.button("Sterge", key=f"del_{profile.name}"):
                delete_profile(profile.name)
                st.rerun()
    else:
        st.info("Niciun profil salvat. Adauga unul mai jos.")

    st.divider()
    st.subheader("Adauga profil nou")
    new_name = st.text_input("Nume profil", placeholder="Ex.: Vocea mea principala")
    new_lang = st.selectbox("Limba profil", ["English", "Russian", "French", "Spanish", "German"], key="profile_lang")
    new_file = st.file_uploader("Inregistrare de referinta (WAV/MP3/FLAC, 10-30 sec)", type=["wav", "mp3", "flac"], key="profile_upload")
    if st.button("Salveaza profilul") and new_name and new_file:
        save_profile(new_name, new_file.getvalue(), new_file.name, new_lang)
        st.success(f"Profil '{new_name}' salvat.")
        st.rerun()

# ===========================================================================
# PAGE: History
# ===========================================================================
elif page == "Istoric":
    st.title("Istoric generari")
    st.caption("Toate run-urile anterioare din outputs/.")

    if not OUTPUTS.exists() or not any(OUTPUTS.iterdir()):
        st.info("Nicio generare anterioara. Genereaza un videoclip pentru a-l vedea aici.")
    else:
        runs = sorted(
            [d for d in OUTPUTS.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        for run_dir in runs:
            video = run_dir / "final_video.mp4"
            script_file = run_dir / "script.json"
            timeline = run_dir / "timeline.json"
            srt = run_dir / "subtitles.srt"

            with st.expander(f"📹 {run_dir.name}", expanded=False):
                if video.exists():
                    st.video(str(video))
                    with video.open("rb") as vf:
                        st.download_button("Descarca MP4", vf, file_name=video.name, mime="video/mp4", key=f"dl_{run_dir.name}")
                else:
                    st.warning("Video final negasit (generare incompleta?).")

                cols = st.columns(3)
                if script_file.exists():
                    with cols[0]:
                        script_data = json.loads(script_file.read_text(encoding="utf-8"))
                        st.metric("Scene", len(script_data.get("scenes", [])))
                        total_beats = sum(len(s.get("beats", [])) for s in script_data.get("scenes", []))
                        st.metric("Beats", total_beats)
                if timeline.exists():
                    with cols[1]:
                        tl_data = json.loads(timeline.read_text(encoding="utf-8"))
                        beats_list = tl_data.get("beats", [])
                        sources = {}
                        for b in beats_list:
                            src = b.get("actual_source", "unknown")
                            sources[src] = sources.get(src, 0) + 1
                        for src, count in sorted(sources.items()):
                            st.metric(f"Vizual: {src}", count)
                if srt.exists():
                    with cols[2]:
                        st.caption("Subtitrari disponibile")
                        st.download_button("Descarca SRT", srt.read_text(encoding="utf-8"), file_name="subtitles.srt", key=f"srt_{run_dir.name}")

                if script_file.exists():
                    with st.expander("Scenariu JSON"):
                        st.json(json.loads(script_file.read_text(encoding="utf-8")))

                if st.button("Sterge acest run", key=f"rmrun_{run_dir.name}"):
                    shutil.rmtree(run_dir, ignore_errors=True)
                    st.rerun()

# ===========================================================================
# PAGE: Generate (main)
# ===========================================================================
else:
    st.title("Autonomous AI Video Generator")
    st.caption("Gemini + local Chatterbox/Piper TTS + Pexels/Pollinations + optional ComfyUI.")
    st.caption("Timeline vizual: 40% ComfyUI, 35% Pexels, 25% imagini AI. Fara loop video artificial.")

    topic = st.text_area("Tema videoclipului", placeholder="Ex.: Cum a schimbat Imperiul Roman comertul in Europa", height=120)
    language = st.selectbox("Limba nararii", ["Romanian", "English", "Russian", "French", "Spanish", "German"])
    aspect_ratio = st.radio("Format video", ["16:9 Long-Form", "9:16 Shorts"], horizontal=True)
    smoke_test = st.checkbox("Smoke test rapid (~1 min, primele cateva beats)", value=False, help="Valideaza intreg pipeline-ul rapid si ieftin inainte de un video complet.")
    duration = 1 if smoke_test else st.slider("Durata tinta (minute)", min_value=10, max_value=20, value=12)
    tts_label = st.selectbox("Motor TTS", available_engines(language))

    # --- Voice reference: saved profile or upload ---
    reference_audio = None
    reference_path = None
    if engine_from_label(tts_label) == "chatterbox":
        profiles = [p for p in list_profiles() if p.language == language]
        source_options = ["Incarca fisier nou"] + [f"Profil: {p.name}" for p in profiles]
        voice_source = st.selectbox("Sursa vocala", source_options)
        if voice_source == "Incarca fisier nou":
            reference_audio = st.file_uploader("Inregistrare de referinta (WAV/MP3/FLAC, 10-30 sec)", type=["wav", "mp3", "flac"])
        else:
            profile_name = voice_source.removeprefix("Profil: ")
            matched = next((p for p in profiles if p.name == profile_name), None)
            if matched:
                reference_path = matched.audio_path
                st.audio(str(reference_path))
                st.caption(f"Folosesc profilul salvat: {matched.name}")

    engine = engine_from_label(tts_label)
    has_reference = reference_audio is not None or reference_path is not None
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
            # Resolve reference audio: saved profile takes priority, then fresh upload.
            if reference_path is None and reference_audio is not None:
                reference_path = run_dir / "reference" / reference_audio.name
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                reference_path.write_bytes(reference_audio.getvalue())

            tts_config = TTSConfig(engine=engine_from_label(tts_label), language=language, reference_audio=reference_path)
            beats = [beat for scene in script.scenes for beat in scene.beats]
            if smoke_test:
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

            # Offer to save the voice reference as a profile for next time.
            if reference_audio is not None and engine_from_label(tts_label) == "chatterbox":
                st.divider()
                save_name = st.text_input("Salveaza aceasta voce ca profil (optional)", key="save_voice_name")
                if st.button("Salveaza profil vocal") and save_name:
                    save_profile(save_name, reference_audio.getvalue(), reference_audio.name, language)
                    st.success(f"Profil '{save_name}' salvat. Il gasesti in pagina Profiluri vocale.")

            with st.expander("Scenariu generat"):
                st.json(script.model_dump())
        except Exception as exc:
            progress.empty()
            status.error(f"Generarea s-a oprit: {exc}")
            if run_dir.exists() and not any(run_dir.iterdir()):
                shutil.rmtree(run_dir)
