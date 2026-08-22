import os
import gradio as gr

# ─── Import FastAPI app (registers all routes: /api/health, /api/generate-image, /) ─
# We import `app` from main.py which has all the FastAPI routes + static file serving.
# We then pass it as `app_kwargs` to demo.launch so Gradio wraps it as the root ASGI app.
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
# This is the correct pattern for HF Gradio SDK + FastAPI:
# gr.mount_gradio_app mounts the Gradio Blocks into the existing FastAPI app.
# The resulting `app` is an ASGI app that:
#   • Serves the custom HTML/JS/CSS UI at "/"
#   • Serves API routes at "/api/*"
#   • Serves the Gradio UI at "/gradio"
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# ─── Launch ──────────────────────────────────────────────────────────────────
# HF Gradio SDK runs `python app.py` as a script.
# We use uvicorn to serve the `app` ASGI object (FastAPI + Gradio mounted).
# This is the correct approach when using gr.mount_gradio_app.
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
