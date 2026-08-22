import os
import gradio as gr

# ─── Import FastAPI app (registers all routes: /api/health, /api/generate-image, /) ─
from main import app as fastapi_app

# ─── Gradio UI (HF Spaces entry point) ───────────────────────────────────────
# Minimal Gradio Blocks — the real custom glassmorphism UI is served at "/" via FastAPI.
demo = gr.Blocks(title="Chitraya AI - FLUX.1 Schnell Image Generator")

with demo:
    gr.Markdown("## 🎨 Chitraya AI — LangChain + FLUX.1 Schnell")
    gr.Markdown(
        "Your full **glassmorphism UI** is served at the **root path `/`** of this Space.\n\n"
        "> Click the link icon (🔗) at the top right of this Space to open the full custom UI."
    )

# ─── Mount Gradio at /gradio on the FastAPI app ──────────────────────────────
# gr.mount_gradio_app mounts the Gradio Blocks into the existing FastAPI app.
# HF Gradio SDK detects the `app` ASGI object and serves it on port 7860.
# DO NOT call uvicorn.run() here — HF handles the server launch automatically.
# For local development, run: uvicorn app:app --host 0.0.0.0 --port 7860
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
