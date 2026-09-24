# Handoff: Autonomous AI Video Generator

## Rules that cannot be broken

- Never read, print, edit, commit, upload, or expose `.env` or API keys.
- Do not delete user files, models, ComfyUI installations, or generated output without explicit approval.
- Do not use artificial video loops. Every narration beat must have a unique visual asset. A fallback must be a new asset, not a repeated clip.
- Keep the application local-first and cost-free apart from user-provided free-tier API keys.
- Make small, reviewable changes. Run focused tests after each change.

## Project location and runtime

- Workspace: `C:\Users\negar\OneDrive\Документы\AI Video maker`
- Windows, Python 3.11 virtual environment: `.venv`
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU; PyTorch CUDA is verified in `.venv`.
- FFmpeg is installed and available on PATH.
- Launch the app with: `streamlit run app.py`

## Current integrations

- Gemini structured script: `script_engine.py`
- Local TTS: `tts_engine.py`
  - Chatterbox V3: English, Russian, French, Spanish, German; requires a 10-30 second consented voice-reference audio file.
  - Piper: Romanian fallback; paths are configured by environment variables once model files are installed.
  - Edge is an online fallback only.
- Visual acquisition and ComfyUI client: `media_engine.py`
- Composition: `video_composer.py`
- Artifacts: `project_artifacts.py` writes `subtitles.srt` and `timeline.json`.

## Timeline architecture

- Gemini returns scenes containing 6-16 short narration beats.
- Each beat has narration text, Pexels keywords, an AI fallback prompt, and a ComfyUI candidate flag.
- TTS produces one audio file per beat.
- Media mix target: 40% ComfyUI, 35% Pexels, 25% AI images.
- Video assets must be at least as long as their narration beat; the composer must trim only, never loop.

## Current external-system state

- `.env` exists locally and has Gemini/Pexels keys. Do not inspect it.
- Comfy Desktop is installed. Its updater fails with a `pygit2_compat` null-byte error; do not repair or reinstall it without asking the user.
- The currently installed ComfyUI instance launches successfully.
- Wan 2.1 Text-to-Video 1.3B workflow is loaded and successfully generated a 2-second fox-in-snow video.
- The next ComfyUI work is to test a 4-second output, then save the workflow in **API format** into `config/wan_api.json` and configure the application to use it.

## Work that remains

1. Add robust preflight checks for `.env` presence (without printing values), FFmpeg, GPU, ComfyUI health, workflow path, and selected TTS engine.
2. Finish the ComfyUI workflow handoff and API-format setup.
3. Add retry/fallback behavior per beat, asset de-duplication, and a persistent voice-profile manager.
4. Implement a 1-minute end-to-end smoke test before attempting a 10-20 minute video.
5. Add a project history page and beat-level retry controls in Streamlit.

## First task for the agent

Read this file and all Python files. Do not modify anything yet. Produce a short implementation plan and identify defects or incompatibilities in the existing code. Wait for user approval before making broad changes.
