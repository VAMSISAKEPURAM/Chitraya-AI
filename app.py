import os
import spaces
import gradio as gr

from backend.utils.config import settings
from backend.agent.image_agent import image_agent
from backend.services.huggingface_service import HuggingFaceServiceError

# ─── Core Generation Function (ZeroGPU-decorated) ────────────────────────────
@spaces.GPU
def generate_image(prompt: str):
    """
    Main generation pipeline:
    User Prompt → Groq LLM Enhancement → FLUX.1 Schnell → Base64 Image
    This function is decorated with @spaces.GPU for ZeroGPU free tier compatibility.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt before generating.")

    clean_prompt = prompt.strip()

    try:
        result = image_agent.generate(clean_prompt)
    except HuggingFaceServiceError as e:
        raise gr.Error(f"Image generation failed: {e.message}")
    except Exception as e:
        raise gr.Error(f"Unexpected error: {str(e)}")

    # Parse base64 data URI → raw bytes for Gradio Image component
    data_uri = result["image"]
    if data_uri.startswith("data:image"):
        import base64, io
        from PIL import Image
        header, b64data = data_uri.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        image = Image.open(io.BytesIO(img_bytes))
    else:
        raise gr.Error("Invalid image response from generation service.")

    return (
        image,
        result["enhanced_prompt"],
        result["original_prompt"],
    )

def get_status():
    hf = "✅ Ready" if settings.is_hf_configured() else "❌ HF_TOKEN missing"
    groq = "✅ Active" if settings.is_groq_configured() else "⚠️ Fallback mode (no Groq key)"
    return f"**HF Inference API:** {hf}  |  **Groq LLM:** {groq}  |  **Model:** `{settings.IMAGE_MODEL}`"

# ─── Gradio UI ────────────────────────────────────────────────────────────────
EXAMPLE_PROMPTS = [
    "A realistic Indian farmer working in a smart agricultural field with golden hour lighting",
    "A futuristic cyberpunk city at sunset with neon reflections and flying cars",
    "A luxury black car in a cinematic dark studio with dramatic rim lighting",
    "A cute astronaut cat walking on mars with Earth in the background, digital painting",
]

theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_700",
    input_background_fill="*neutral_800",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed, #a855f7)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6d28d9, #9333ea)",
    button_primary_text_color="white",
)

css = """
#title { text-align: center; }
#title h1 { 
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0;
}
#title p { color: #94a3b8; margin-top: 4px; }
#status-box { border-radius: 10px; padding: 10px; background: #1e293b; font-size: 0.9rem; }
#gen-btn { height: 52px; font-size: 1.1rem; font-weight: 700; }
#output-image { border-radius: 12px; }
.prompt-compare { background: #1e293b; border-radius: 10px; padding: 12px; font-size: 0.85rem; }
footer { display: none !important; }
"""

with gr.Blocks(theme=theme, css=css, title="Chitraya AI - FLUX.1 Schnell Image Generator") as demo:

    # Header
    with gr.Column(elem_id="title"):
        gr.HTML("<h1>🎨 Chitraya AI</h1><p>LangChain + Groq Prompt Engineering · FLUX.1 Schnell · ZeroGPU</p>")

    # Status bar
    status_md = gr.Markdown(get_status(), elem_id="status-box")

    with gr.Row():
        # Left panel: Input
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="✏️ Describe your image",
                placeholder="e.g. A futuristic cyberpunk city at sunset with neon reflections...",
                lines=4,
                max_lines=6,
                elem_id="prompt-input"
            )

            gr.Examples(
                examples=EXAMPLE_PROMPTS,
                inputs=prompt_input,
                label="💡 Quick Examples — click to use",
                examples_per_page=4,
            )

            generate_btn = gr.Button(
                "✨ Generate Image",
                variant="primary",
                elem_id="gen-btn"
            )

            with gr.Accordion("📝 Prompt Details", open=False):
                original_prompt_box = gr.Textbox(
                    label="Your Original Prompt",
                    interactive=False,
                    lines=2,
                )
                enhanced_prompt_box = gr.Textbox(
                    label="🤖 Groq-Enhanced Prompt (sent to FLUX.1)",
                    interactive=False,
                    lines=4,
                )

        # Right panel: Output
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="🖼️ Generated Image",
                type="pil",
                elem_id="output-image",
                height=512,
                show_download_button=True,
            )

    # Wire up the generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input],
        outputs=[output_image, enhanced_prompt_box, original_prompt_box],
        api_name="generate",
    )

    # Also allow pressing Enter in the textbox
    prompt_input.submit(
        fn=generate_image,
        inputs=[prompt_input],
        outputs=[output_image, enhanced_prompt_box, original_prompt_box],
    )

    gr.Markdown(
        "---\n*Powered by LangChain · Groq LLaMA-3 · FLUX.1 Schnell via HF Inference API · Gradio · ZeroGPU*",
        elem_id="footer-note"
    )

# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
