# Autonomous Long-Form AI Video Generator

Local, zero-cost-oriented Streamlit pipeline for narrated YouTube videos. It combines Gemini for a structured script, local Chatterbox/Piper narration, Pexels b-roll, Pollinations image fallbacks, optional local ComfyUI video, and MoviePy/FFmpeg composition.

## Setup (Windows / Python 3.10+)

1. Install FFmpeg and make `ffmpeg` available on `PATH`.
2. Create and activate the virtual environment:

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and enter your Gemini and Pexels API keys.
4. Optional, for local AI action shots: start ComfyUI with API access and export its workflow in **API format**. Set `COMFYUI_WORKFLOW_PATH` to that JSON file. The workflow must include a `CLIPTextEncode` node; its positive prompt is replaced automatically.
5. Start the UI:

   ```powershell
   streamlit run app.py
   ```

Generated assets and the final MP4 are saved under `outputs/`. Pexels downloads and Pollinations images are external services; review their current terms and content licenses before publishing.

## Visual timeline

Gemini generates short narration beats rather than one long visual per scene. Each beat receives its own narration audio and a unique visual asset. The default mix is 40% ComfyUI, 35% Pexels, and 25% AI images. Video assets are never looped to fill narration; an asset that is too short falls back to a new Pexels clip or image. Each completed run includes `timeline.json` and `subtitles.srt` next to the final MP4.

## Local narration

The application has three selectable engines:

- **Chatterbox V3 (local):** recommended for English, Russian, French, Spanish, and German. It requires a 10-30 second voice-reference WAV, MP3, or FLAC file. Install the source version you downloaded with:

  ```powershell
  cd "C:\Users\negar\Downloads\chatterbox-master\chatterbox-master"
  py -3.10 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .
  ```

  To make it available to this application, install the same package into the project's virtual environment too:

  ```powershell
  cd "C:\Users\negar\OneDrive\Документы\AI Video maker"
  .\.venv\Scripts\Activate.ps1
  pip install -e "C:\Users\negar\Downloads\chatterbox-master\chatterbox-master"
  ```

  The default package index can install CPU-only PyTorch. For an NVIDIA RTX GPU, replace it with the CUDA build required by Chatterbox before running the app:

  ```powershell
  pip uninstall -y torch torchvision torchaudio
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
  ```

  The final command must print `True` and the NVIDIA GPU name.

- **Piper (local):** use it for Romanian. Download a model `.onnx` and its matching `.onnx.json` config, then set `PIPER_MODEL_PATH_RO` and `PIPER_CONFIG_PATH_RO` in `.env`.
- **Edge (online fallback):** remains available only as a fallback when a local engine is not installed.

Chatterbox's current multilingual model does not officially list Romanian; the UI therefore does not offer it for Romanian narration.
