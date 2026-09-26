from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

from media_engine import VisualSource, audio_duration, acquire_visual, build_scene_media
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
# Optional password gate (set APP_PASSWORD in .env to enable)
# ---------------------------------------------------------------------------
_app_password = os.environ.get("APP_PASSWORD", "").strip()
if _app_password:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        st.title("Autentificare")
        pwd = st.text_input("Parola", type="password")
        if st.button("Intra"):
            if pwd == _app_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Parola incorecta.")
        st.stop()

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
        try:
            save_profile(new_name, new_file.getvalue(), new_file.name, new_lang)
            st.success(f"Profil '{new_name}' salvat.")
            st.rerun()
        except ValueError as ve:
            st.error(str(ve))

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

                # --- Run metadata & duration report ---
                meta_file = run_dir / "run_meta.json"
                meta = {}
                cols = st.columns(4)
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    target_min = meta.get("target_minutes", "?")
                    actual_min = round(meta.get("actual_seconds", 0) / 60, 1)
                    cols[0].metric("Durata tinta", f"{target_min} min")
                    cols[1].metric("Durata reala", f"{actual_min} min")
                    cols[2].metric("Beats", meta.get("beats_total", "?"))
                    cols[3].metric("Motor TTS", meta.get("tts_engine", "?"))

                if timeline.exists():
                    tl_data = json.loads(timeline.read_text(encoding="utf-8"))
                    beats_list = tl_data.get("beats", [])
                    sources = {}
                    for b in beats_list:
                        src = b.get("actual_source", "unknown")
                        sources[src] = sources.get(src, 0) + 1
                    src_cols = st.columns(len(sources) + 1)
                    for idx, (src, count) in enumerate(sorted(sources.items())):
                        src_cols[idx].metric(f"Vizual: {src}", count)

                    # --- Beat-level detail with retry controls ---
                    with st.expander("Detalii per beat (retry manual)"):
                        for beat_entry in beats_list:
                            bid = beat_entry.get("beat_id", "?")
                            narr = beat_entry.get("narration_text", "")[:80]
                            planned = beat_entry.get("planned_source", "?")
                            actual = beat_entry.get("actual_source", "?")
                            vkind = beat_entry.get("visual_kind", "?")
                            vpath = beat_entry.get("visual_path", "")

                            bc1, bc2, bc3 = st.columns([5, 2, 2])
                            bc1.write(f"**Beat {bid}**: {narr}...")
                            bc2.write(f"📋 {planned} → ✅ {actual} ({vkind})")

                            retry_key = f"retry_{run_dir.name}_{bid}"
                            new_kw_key = f"kw_{run_dir.name}_{bid}"
                            new_keywords = bc3.text_input("Keywords", value=beat_entry.get("narration_text", "")[:30], key=new_kw_key, label_visibility="collapsed")
                            if st.button(f"Re-genera vizual beat {bid}", key=retry_key):
                                from script_engine import NarrationBeat
                                retry_beat = NarrationBeat(
                                    beat_id=bid,
                                    narration_text=beat_entry.get("narration_text", "placeholder narration"),
                                    pexels_keywords=new_keywords,
                                    fallback_ai_prompt=beat_entry.get("narration_text", "cinematic scene"),
                                    is_key_action_moment=False,
                                )
                                audio_path = Path(beat_entry.get("audio_path", ""))
                                visuals_dir = run_dir / "visuals"
                                try:
                                    ar = meta.get("aspect_ratio", "16:9 Long-Form") if meta_file.exists() else "16:9 Long-Form"
                                    dur = audio_duration(audio_path) if audio_path.exists() else 5.0
                                    new_path, new_kind, new_source = acquire_visual(
                                        retry_beat, VisualSource.PEXELS, visuals_dir, ar, dur,
                                    )
                                    # Update timeline entry.
                                    beat_entry["visual_path"] = str(new_path)
                                    beat_entry["visual_kind"] = new_kind
                                    beat_entry["actual_source"] = new_source
                                    timeline.write_text(json.dumps(tl_data, ensure_ascii=False, indent=2), encoding="utf-8")
                                    # Re-compose the full video with updated visuals.
                                    from media_engine import SceneMedia
                                    recompose_media = []
                                    for b in beats_list:
                                        from script_engine import NarrationBeat
                                        rb = NarrationBeat(
                                            beat_id=b["beat_id"],
                                            narration_text=b.get("narration_text", ""),
                                            pexels_keywords=b.get("narration_text", "")[:30],
                                            fallback_ai_prompt=b.get("narration_text", ""),
                                            is_key_action_moment=False,
                                        )
                                        recompose_media.append(SceneMedia(
                                            beat=rb,
                                            audio_path=Path(b["audio_path"]),
                                            visual_path=Path(b["visual_path"]),
                                            visual_kind=b["visual_kind"],
                                            planned_source=b.get("planned_source", ""),
                                            actual_source=b["actual_source"],
                                        ))
                                    ar = meta.get("aspect_ratio", "16:9 Long-Form") if meta_file.exists() else "16:9 Long-Form"
                                    st.info("Re-randare video cu vizualul actualizat...")
                                    compose_video(recompose_media, video, ar)
                                    st.success(f"Beat {bid} re-generat si video-ul recompus.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Retry beat {bid} a esuat: {exc}")

                if srt.exists():
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
            log.info("Starting generation: topic=%.80s lang=%s ar=%s smoke=%s", topic, language, aspect_ratio, smoke_test)
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
            # Save duration metadata alongside the timeline for the history page.
            total_audio_sec = sum(audio_duration(ap) for ap in audio_paths)
            write_timeline_manifest(media, run_dir / "timeline.json")
            (run_dir / "run_meta.json").write_text(json.dumps({
                "target_minutes": duration,
                "actual_seconds": round(total_audio_sec, 2),
                "beats_total": len(beats),
                "smoke_test": smoke_test,
                "language": language,
                "aspect_ratio": aspect_ratio,
                "tts_engine": engine,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            progress.progress(70, text="Vizualuri pregatite")

            status.write("4/4 - Randare finala locala. Aceasta poate dura cateva minute...")
            output = compose_video(media, run_dir / "final_video.mp4", aspect_ratio)
            progress.progress(100, text="Videoclip finalizat")
            # --- Duration report (total_audio_sec computed earlier for run_meta.json) ---
            total_audio_min = total_audio_sec / 60
            target_min = duration
            delta = total_audio_min - target_min
            delta_label = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

            status.success(f"Gata: {output.name}")
            st.video(str(output))

            report_cols = st.columns(3)
            report_cols[0].metric("Durata tinta", f"{target_min} min")
            report_cols[1].metric("Durata reala", f"{total_audio_min:.1f} min", delta=f"{delta_label} min")
            report_cols[2].metric("Beats randate", f"{len(beats)}")

            with output.open("rb") as video_file:
                st.download_button("Descarca MP4", video_file, file_name=output.name, mime="video/mp4")

            # Optional YouTube upload.
            from youtube_upload import is_available as yt_available
            if yt_available():
                st.divider()
                yt_title = st.text_input("Titlu YouTube", value=script.title[:100], key="yt_title")
                yt_privacy = st.selectbox("Vizibilitate", ["private", "unlisted", "public"], key="yt_privacy")
                if st.button("Urca pe YouTube", key="yt_upload"):
                    try:
                        from youtube_upload import upload_video
                        vid_id = upload_video(output, yt_title, description=topic, privacy=yt_privacy)
                        st.success(f"Video urcat pe YouTube: https://youtu.be/{vid_id}")
                        log.info("YouTube upload: https://youtu.be/%s", vid_id)
                    except Exception as yt_exc:
                        log.exception("YouTube upload failed")
                        st.error(f"Upload YouTube esuat: {yt_exc}")

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
            log.exception("Generation failed for topic: %.80s", topic)
            progress.empty()
            status.error(f"Generarea s-a oprit: {exc}")
            if run_dir.exists() and not any(run_dir.iterdir()):
                shutil.rmtree(run_dir)
