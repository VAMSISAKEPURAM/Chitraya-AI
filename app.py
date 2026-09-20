import os
import io
import base64
import logging
import gradio as gr
from PIL import Image

from backend.utils.config import settings
from backend.agent.image_agent import image_agent
from backend.services.huggingface_service import HuggingFaceServiceError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("chitraya_app")

try:
    import spaces
    @spaces.GPU
    def _zerogpu_init():
        """ZeroGPU startup validation function."""
        return True
except Exception:
    def _zerogpu_init():
        return False

# ─── Handlers ─────────────────────────────────────────────────────────────────
def generate_from_text(prompt: str):
    """Generates an AI image from a raw text prompt."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter or select a prompt before clicking Generate.")

    clean_prompt = prompt.strip()
    logger.info(f"Text-to-image request: '{clean_prompt[:80]}...'")

    try:
        result = image_agent.generate(clean_prompt)
    except HuggingFaceServiceError as e:
        raise gr.Error(f"Hugging Face API Error: {e.message}")
    except Exception as e:
        raise gr.Error(f"Generation Error: {str(e)}")

    # Parse image
    data_uri = result.get("image", "")
    if isinstance(data_uri, str) and data_uri.startswith("data:image"):
        header, b64data = data_uri.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(b64data)))
    elif isinstance(data_uri, Image.Image):
        image = data_uri
    else:
        raise gr.Error("Failed to parse image from generator output.")

    model_info = result.get("model", settings.IMAGE_MODEL)
    return image, clean_prompt, f"⚡ {model_info}"


def transform_uploaded_photo(input_image, transform_prompt: str, preset_style: str = ""):
    """Takes an uploaded photo and applies custom visual edits or style transformations."""
    if input_image is None:
        raise gr.Error("Please upload a photo first before transforming.")

    clean_prompt = transform_prompt.strip() if transform_prompt else ""
    if not clean_prompt and not preset_style:
        raise gr.Error("Please describe the changes you want or click a style preset button below.")

    logger.info(f"Photo transform request with prompt: '{clean_prompt}', preset: '{preset_style}'")

    try:
        result = image_agent.transform_image(
            input_image=input_image,
            transform_prompt=clean_prompt,
            style_preset=preset_style
        )
    except HuggingFaceServiceError as e:
        raise gr.Error(f"Hugging Face API Error: {e.message}")
    except Exception as e:
        raise gr.Error(f"Transformation Error: {str(e)}")

    # Parse output image
    data_uri = result.get("image", "")
    if isinstance(data_uri, str) and data_uri.startswith("data:image"):
        header, b64data = data_uri.split(",", 1)
        output_img = Image.open(io.BytesIO(base64.b64decode(b64data)))
    elif isinstance(data_uri, Image.Image):
        output_img = data_uri
    else:
        raise gr.Error("Failed to parse transformed image.")

    model_info = result.get("model", settings.IMAGE_MODEL)
    effective_prompt = result.get("prompt", clean_prompt)
    return output_img, effective_prompt, f"⚡ {model_info}"


def get_status():
    hf_ok = settings.is_hf_configured()
    hf_icon = "🟢 Ready" if hf_ok else "🔴 Missing Secret (HF_TOKEN)"
    return f"**Hugging Face Inference:** {hf_icon} &nbsp;|&nbsp; **Engine:** FLUX.1 Schnell &nbsp;|&nbsp; **Features:** Text-to-Image + Photo Remix"

# ─── Preset Prompts & Transformation Styles ───────────────────────────────────
TEXT_PROMPT_1 = "A realistic Indian farmer working in a smart agricultural field with golden hour lighting"
TEXT_PROMPT_2 = "A futuristic cyberpunk city at sunset with neon reflections and flying autonomous vehicles"
TEXT_PROMPT_3 = "A luxury black sports car in a cinematic dark studio with dramatic neon rim lighting"
TEXT_PROMPT_4 = "A cute astronaut cat walking on Mars with Earth visible in the star-filled sky, digital painting"

STYLE_CYBERPUNK = "Reimagine as a high-tech Cyberpunk 2077 warrior with glowing neon cybernetic implants, holographic visor, in a rainy futuristic city with reflections"
STYLE_ANIME = "Transform into a masterwork Studio Ghibli / Makoto Shinkai anime aesthetic, lush hand-drawn painterly textures, vibrant whimsical sky"
STYLE_PIXAR = "Transform into a charming 3D Disney Pixar animated character, expressive eyes, smooth stylized 3D CGI rendering, octane lighting"
STYLE_ROYAL = "Paint as a majestic 17th-century Renaissance oil painting, dressed in royal velvet attire with ornate gold embroidery, dramatic chiaroscuro Rembrandt lighting"
STYLE_SUPERHERO = "Transform into a cinematic comic-book superhero in dynamic battle armor, surrounded by glowing energy particles and dramatic atmosphere"
STYLE_VINTAGE = "Reimagine as a 1990s vintage 35mm analog film photograph, Kodak Portra colors, soft film grain, authentic retro nostalgic lighting"

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
#title { text-align: center; margin-bottom: 8px; }
#title h1 { 
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 2px;
}
#title p { color: #94a3b8; font-size: 0.95rem; }
#status-box { border-radius: 8px; padding: 8px 14px; background: #1e293b; font-size: 0.88rem; }
.action-btn { height: 50px; font-size: 1.1rem; font-weight: 700; margin-top: 10px; }
.output-preview { border-radius: 12px; }
.quick-chip { margin: 2px 0; }
"""

import inspect

blocks_params = inspect.signature(gr.Blocks.__init__).parameters
launch_params = inspect.signature(gr.Blocks.launch).parameters

blocks_kwargs = {"title": "Chitraya AI - Studio Image Generator & Photo Remix"}
launch_kwargs = {
    "server_name": "0.0.0.0",
    "server_port": int(os.getenv("PORT", 7860)),
    "show_error": True,
}

if "ssr_mode" in launch_params:
    launch_kwargs["ssr_mode"] = False

if "theme" in blocks_params:
    blocks_kwargs["theme"] = theme
elif "theme" in launch_params:
    launch_kwargs["theme"] = theme

if "css" in blocks_params:
    blocks_kwargs["css"] = css
elif "css" in launch_params:
    launch_kwargs["css"] = css

with gr.Blocks(**blocks_kwargs) as demo:

    # Header & Status
    with gr.Column(elem_id="title"):
        gr.HTML("<h1>🎨 Chitraya AI Studio</h1><p>Text-to-Image Synthesis & Photo Remix / Transformation · FLUX.1 Schnell</p>")

    status_md = gr.Markdown(get_status(), elem_id="status-box")

    with gr.Tabs():
        
        # ─── TAB 1: Text to Image ─────────────────────────────────────────────
        with gr.TabItem("🎨 Text to Image", id="tab_text2img"):
            with gr.Row():
                with gr.Column(scale=1):
                    txt_prompt_input = gr.Textbox(
                        label="✏️ Describe the image you want to create",
                        placeholder="Type your prompt here or click any preset below...",
                        lines=4,
                        max_lines=6,
                        elem_id="txt-prompt-input"
                    )

                    gr.Markdown("**💡 Quick Inspiration Prompts (click to set):**")
                    with gr.Column():
                        t_btn1 = gr.Button(f"🌾 {TEXT_PROMPT_1}", size="sm", elem_classes=["quick-chip"])
                        t_btn2 = gr.Button(f"🌃 {TEXT_PROMPT_2}", size="sm", elem_classes=["quick-chip"])
                        t_btn3 = gr.Button(f"🏎️ {TEXT_PROMPT_3}", size="sm", elem_classes=["quick-chip"])
                        t_btn4 = gr.Button(f"🚀 {TEXT_PROMPT_4}", size="sm", elem_classes=["quick-chip"])

                    txt_generate_btn = gr.Button(
                        "✨ Generate Image",
                        variant="primary",
                        elem_classes=["action-btn"]
                    )

                    with gr.Accordion("📋 Prompt & Model Details", open=False):
                        txt_model_box = gr.Textbox(label="⚡ Model Engine Used", interactive=False, lines=1)
                        txt_sent_box = gr.Textbox(label="Prompt Sent to Model", interactive=False, lines=2)

                with gr.Column(scale=1):
                    txt_output_image = gr.Image(
                        label="🖼️ Generated AI Artwork",
                        type="pil",
                        elem_classes=["output-preview"],
                        height=500,
                    )

            # Wire up Text-to-Image buttons
            t_btn1.click(fn=lambda: TEXT_PROMPT_1, inputs=[], outputs=[txt_prompt_input])
            t_btn2.click(fn=lambda: TEXT_PROMPT_2, inputs=[], outputs=[txt_prompt_input])
            t_btn3.click(fn=lambda: TEXT_PROMPT_3, inputs=[], outputs=[txt_prompt_input])
            t_btn4.click(fn=lambda: TEXT_PROMPT_4, inputs=[], outputs=[txt_prompt_input])

            txt_generate_btn.click(
                fn=generate_from_text,
                inputs=[txt_prompt_input],
                outputs=[txt_output_image, txt_sent_box, txt_model_box],
                api_name="generate_text",
            )
            txt_prompt_input.submit(
                fn=generate_from_text,
                inputs=[txt_prompt_input],
                outputs=[txt_output_image, txt_sent_box, txt_model_box],
            )

        # ─── TAB 2: Photo Remix & Transform ───────────────────────────────────
        with gr.TabItem("📸 Photo Remix & Transform", id="tab_img2img"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input_image = gr.Image(
                        label="📤 Upload Your Photo (Portrait, Pet, Landscape, Selfie)",
                        type="pil",
                        height=280,
                    )

                    transform_prompt_input = gr.Textbox(
                        label="✨ Describe Changes or Custom Style",
                        placeholder="e.g. Turn into a cyberpunk warrior with glowing tattoos, wearing samurai armor, sunset in Tokyo...",
                        lines=3,
                        max_lines=5,
                    )

                    gr.Markdown("**⚡ 1-Click Style Transformation Presets (click to apply):**")
                    with gr.Row():
                        s_btn1 = gr.Button("⚡ Cyberpunk 2077", size="sm")
                        s_btn2 = gr.Button("🎌 Anime / Ghibli", size="sm")
                        s_btn3 = gr.Button("🎬 3D Pixar Animation", size="sm")
                    with gr.Row():
                        s_btn4 = gr.Button("👑 Royal Oil Painting", size="sm")
                        s_btn5 = gr.Button("🦸 Cinematic Superhero", size="sm")
                        s_btn6 = gr.Button("📼 90s Vintage Film", size="sm")

                    transform_btn = gr.Button(
                        "🪄 Transform Photo",
                        variant="primary",
                        elem_classes=["action-btn"]
                    )

                    with gr.Accordion("📋 Transformation Engine Details", open=False):
                        trans_model_box = gr.Textbox(label="⚡ Model Engine Used", interactive=False, lines=1)
                        trans_sent_box = gr.Textbox(label="Synthesized Transformation Prompt", interactive=False, lines=3)

                with gr.Column(scale=1):
                    transformed_output_image = gr.Image(
                        label="🖼️ Transformed AI Artwork",
                        type="pil",
                        elem_classes=["output-preview"],
                        height=500,
                    )

            # Preset click handlers
            s_btn1.click(fn=lambda: STYLE_CYBERPUNK, inputs=[], outputs=[transform_prompt_input])
            s_btn2.click(fn=lambda: STYLE_ANIME, inputs=[], outputs=[transform_prompt_input])
            s_btn3.click(fn=lambda: STYLE_PIXAR, inputs=[], outputs=[transform_prompt_input])
            s_btn4.click(fn=lambda: STYLE_ROYAL, inputs=[], outputs=[transform_prompt_input])
            s_btn5.click(fn=lambda: STYLE_SUPERHERO, inputs=[], outputs=[transform_prompt_input])
            s_btn6.click(fn=lambda: STYLE_VINTAGE, inputs=[], outputs=[transform_prompt_input])

            # Wire up Transform Button & Enter key
            transform_btn.click(
                fn=transform_uploaded_photo,
                inputs=[upload_input_image, transform_prompt_input],
                outputs=[transformed_output_image, trans_sent_box, trans_model_box],
                api_name="transform_photo",
            )
            transform_prompt_input.submit(
                fn=transform_uploaded_photo,
                inputs=[upload_input_image, transform_prompt_input],
                outputs=[transformed_output_image, trans_sent_box, trans_model_box],
            )

    gr.Markdown(
        "---\n*Chitraya AI Studio · Powered by FLUX.1 Schnell on Hugging Face Inference API · Gradio*",
    )

demo.queue(default_concurrency_limit=10)

if __name__ == "__main__":
    demo.launch(**launch_kwargs)



