import os
import gradio as gr

# ─── Import FastAPI app (all routes: /api/health, /api/generate-image, /, /static) ─
from main import app as fastapi_app

# ─── Gradio UI ────────────────────────────────────────────────────────────────
demo = gr.Blocks(title="Chitraya AI - FLUX.1 Schnell Image Generator")

with demo:
    gr.Markdown("## 🎨 Chitraya AI — LangChain + FLUX.1 Schnell")
    gr.Markdown(
        "Your full **glassmorphism UI** is served at the **root path `/`** of this Space.\n\n"
        "> Click the link icon (🔗) at the top right of this Space to open the full custom UI."
    )

# ─── Mount Gradio at /gradio on the FastAPI app ──────────────────────────────
# gr.mount_gradio_app returns an ASGI app combining FastAPI routes + Gradio UI.
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# ─── Launch (blocking) ───────────────────────────────────────────────────────
# IMPORTANT: This must be at module level (NOT inside __main__) for HF Gradio SDK.
# HF runs `python app.py` as a script — demo.launch() keeps the process alive.
# Without this blocking call, the script exits immediately (Exit code: 0 error).
demo.launch(server_name="0.0.0.0", server_port=7860)
