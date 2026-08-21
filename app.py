import os
import uvicorn
import gradio as gr
from main import app as fastapi_app

# Create a Gradio Blocks container
demo = gr.Blocks(title="Chitraya AI - LangChain + FLUX.1 Schnell Generator")

with demo:
    gr.Markdown("# 🎨 Chitraya AI - FLUX.1 Schnell Image Generator")
    gr.Markdown("Your custom web interface is active! Open `/` to view the full glassmorphism application.")

# Mount Gradio onto our FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
