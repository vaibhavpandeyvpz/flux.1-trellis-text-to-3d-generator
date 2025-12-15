# ============================================================================
# Imports
# ============================================================================
import gradio as gr
import spaces
from gradio_litmodel3d import LitModel3D

import os
import shutil

os.environ["SPCONV_ALGO"] = "native"
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
import random
import imageio
from easydict import EasyDict as edict
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

# ============================================================================
# Configuration
# ============================================================================
# Get Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)


# ============================================================================
# Helper Functions for FLUX
# ============================================================================
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """Calculate shift parameter for FLUX scheduler based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve and set timesteps for the scheduler."""
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# ============================================================================
# FLUX Pipeline Function
# ============================================================================
@torch.inference_mode()
def flux_pipe_call_that_returns_an_iterable_of_images(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    good_vae: Optional[Any] = None,
    enable_live_preview: bool = True,
):
    """
    Custom FLUX pipeline function that yields intermediate images during generation.
    This enables live preview functionality.
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device

    # 3. Encode prompt
    lora_scale = (
        joint_attention_kwargs.get("scale", None)
        if joint_attention_kwargs is not None
        else None
    )
    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    self._num_timesteps = len(timesteps)

    # Handle guidance
    guidance = (
        torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(
            latents.shape[0]
        )
        if self.transformer.config.guidance_embeds
        else None
    )

    # 6. Denoising loop
    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        # Yield intermediate result if live preview is enabled
        if enable_live_preview:
            latents_for_image = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents_for_image = (
                latents_for_image / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents_for_image, return_dict=False)[0]
            yield self.image_processor.postprocess(image, output_type=output_type)[0]

        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        torch.cuda.empty_cache()

    # Final image using good_vae
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / good_vae.config.scaling_factor) + good_vae.config.shift_factor
    image = good_vae.decode(latents, return_dict=False)[0]
    self.maybe_free_model_hooks()
    torch.cuda.empty_cache()
    yield self.image_processor.postprocess(image, output_type=output_type)[0]


# ============================================================================
# Helper Functions for TRELLIS
# ============================================================================
def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy(),
            "_features_dc": gs._features_dc.cpu().numpy(),
            "_scaling": gs._scaling.cpu().numpy(),
            "_rotation": gs._rotation.cpu().numpy(),
            "_opacity": gs._opacity.cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device="cuda")
    gs._features_dc = torch.tensor(state["gaussian"]["_features_dc"], device="cuda")
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device="cuda")
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device="cuda")
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device="cuda")

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )

    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """Get the random seed for generation."""
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


# ============================================================================
# Model Loading
# ============================================================================
print("Loading TAEF1 VAE (fast preview)...")
taef1 = AutoencoderTiny.from_pretrained(
    "madebyollin/taef1", torch_dtype=dtype, token=hf_token
).to(device)

print("Loading FLUX.1-dev VAE (high quality)...")
good_vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="vae",
    torch_dtype=dtype,
    token=hf_token,
).to(device)

print("Loading FLUX.1-dev pipeline...")
flux_pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=dtype,
    vae=taef1,
    token=hf_token,
).to(device)

# Attach the custom pipeline function
flux_pipe.flux_pipe_call_that_returns_an_iterable_of_images = (
    flux_pipe_call_that_returns_an_iterable_of_images.__get__(flux_pipe)
)

print("Loading TRELLIS pipeline...")
trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "microsoft/TRELLIS-image-large"
)
trellis_pipeline.cuda()
try:
    trellis_pipeline.preprocess_image(
        Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    )  # Preload background removal
except Exception:
    pass

torch.cuda.empty_cache()


# ============================================================================
# Session Management
# ============================================================================
def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


# ============================================================================
# Step 1: Text to Image Generation
# ============================================================================
@spaces.GPU(duration=75)
def generate_text_to_image(
    prompt,
    seed,
    randomize_seed,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    enable_live_preview,
    use_quality_vae,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate image from text prompt using FLUX.1."""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)

    # Determine which VAE to use for final output
    final_vae = good_vae if use_quality_vae else taef1

    # Generate images
    last_image = None
    for img in flux_pipe.flux_pipe_call_that_returns_an_iterable_of_images(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        output_type="pil",
        good_vae=final_vae,
        enable_live_preview=enable_live_preview,
    ):
        last_image = img
        if enable_live_preview:
            yield img, seed, gr.update(interactive=True, visible=True)

    # Return final image
    if not enable_live_preview or last_image is not None:
        yield last_image, seed, gr.update(interactive=True, visible=True)


# ============================================================================
# Step 2: Image to 3D Generation
# ============================================================================
@spaces.GPU(duration=120)
def generate_3d_model(
    image: Image.Image,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[dict, str, str, str, gr.Button]:
    """Convert image to 3D model and extract GLB file."""
    if image is None:
        return None, None, None, None, gr.update(interactive=False, visible=True)

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

    # Generate 3D model (TRELLIS will handle background removal automatically)
    outputs = trellis_pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=True,  # Let TRELLIS handle background removal
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )

    # Render video
    video = render_utils.render_video(outputs["gaussian"][0], num_frames=120)["color"]
    video_geo = render_utils.render_video(outputs["mesh"][0], num_frames=120)["normal"]
    video = [
        np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))
    ]
    video_path = os.path.join(user_dir, "sample.mp4")
    imageio.mimsave(video_path, video, fps=15)

    # Extract GLB
    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]
    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False
    )
    glb_path = os.path.join(user_dir, "sample.glb")
    glb.export(glb_path)

    # Pack state for optional Gaussian extraction
    state = pack_state(gs, mesh)

    torch.cuda.empty_cache()
    return (
        state,
        video_path,
        glb_path,
        glb_path,
        gr.update(interactive=True, visible=True),
    )


@spaces.GPU
def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """Extract a Gaussian splatting file from the generated 3D model."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, "sample.ply")
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


# ============================================================================
# Gradio UI
# ============================================================================
css = """
#col-container {
    max-width: 1400px;
    margin: 0 auto;
}
"""

examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A delicious ceviche cheesecake slice",
]

with gr.Blocks(css=css, delete_cache=(600, 600)) as demo:
    demo.load(start_session)
    demo.unload(end_session)

    gr.Markdown(
        """
        # 🎨 Text to Image to 3D Generator
        
        Complete workflow: Generate images from text using FLUX.1 → Create 3D models with TRELLIS
        
        **Workflow Steps:**
        1. **Text to Image**: Enter a prompt and generate an image using FLUX.1 [dev]
        2. **3D Generation**: Confirm and generate a 3D model using TRELLIS (background removal is handled automatically)
        
        Each step requires your confirmation before proceeding to the next.
        """
    )

    with gr.Column(elem_id="col-container"):
        # Step 1: Text to Image Generation
        gr.Markdown("## Step 1: Text to Image Generation")
        with gr.Row():
            # Left: Settings
            with gr.Column(scale=1):
                prompt = gr.Text(
                    label="Prompt",
                    show_label=True,
                    max_lines=3,
                    placeholder="Enter your text prompt here...",
                )

                with gr.Row():
                    generate_image_btn = gr.Button(
                        "Generate Image", variant="primary", scale=1
                    )
                    confirm_image_btn = gr.Button(
                        "Confirm & Generate 3D",
                        variant="secondary",
                        scale=1,
                        interactive=False,
                        visible=False,
                    )

                gr.Markdown("### Image Generation Settings")
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=15.0,
                        step=0.1,
                        value=3.5,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )

                enable_live_preview = gr.Checkbox(
                    label="Enable Live Preview",
                    value=True,
                    info="Show intermediate images during generation",
                )
                use_quality_vae = gr.Checkbox(
                    label="Use Quality VAE for Final Output",
                    value=True,
                    info="Use high-quality VAE for final image (slower but better quality)",
                )
                image_seed = gr.Number(
                    label="Seed", value=0, interactive=False, visible=False
                )

            # Right: Preview
            with gr.Column(scale=1):
                generated_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=500,
                )

        # Step 2: 3D Generation
        gr.Markdown("## Step 2: 3D Model Generation")
        with gr.Row():
            # Left: Settings
            with gr.Column(scale=1):
                generate_3d_btn = gr.Button(
                    "Generate 3D Model",
                    variant="primary",
                    interactive=False,
                    visible=False,
                )
                extract_gs_btn = gr.Button(
                    "Extract Gaussian", interactive=False, visible=False
                )

                gr.Markdown("### 3D Generation Settings")
                seed_3d = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed_3d = gr.Checkbox(label="Randomize Seed", value=True)

                gr.Markdown("**Stage 1: Sparse Structure Generation**")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(
                        0.0,
                        10.0,
                        label="Guidance Strength",
                        value=7.5,
                        step=0.1,
                    )
                    ss_sampling_steps = gr.Slider(
                        1, 50, label="Sampling Steps", value=12, step=1
                    )

                gr.Markdown("**Stage 2: Structured Latent Generation**")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(
                        0.0,
                        10.0,
                        label="Guidance Strength",
                        value=3.0,
                        step=0.1,
                    )
                    slat_sampling_steps = gr.Slider(
                        1, 50, label="Sampling Steps", value=12, step=1
                    )

                gr.Markdown("**GLB Extraction Settings**")
                mesh_simplify = gr.Slider(
                    0.9, 0.98, label="Simplify", value=0.95, step=0.01
                )
                texture_size = gr.Slider(
                    512, 2048, label="Texture Size", value=1024, step=512
                )

            # Right: Previews
            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="Generated 3D Asset",
                    autoplay=True,
                    loop=True,
                    height=300,
                    visible=False,
                )
                model_output = LitModel3D(
                    label="Extracted GLB/Gaussian",
                    exposure=10.0,
                    height=300,
                    visible=False,
                )

                with gr.Row():
                    download_glb = gr.DownloadButton(
                        label="Download GLB", interactive=False, visible=False
                    )
                    download_gs = gr.DownloadButton(
                        label="Download Gaussian", interactive=False, visible=False
                    )

    # State variables
    output_buf = gr.State()
    current_image = gr.State()

    # Examples
    gr.Examples(
        examples=examples,
        inputs=[prompt],
    )

    # ========================================================================
    # Event Handlers
    # ========================================================================

    # Step 1: Generate image
    def on_image_generated(img, seed_val):
        if img is not None:
            return (
                img,
                seed_val,
                gr.update(interactive=True, visible=True),  # confirm_image_btn
                img,  # current_image state
            )
        return (
            None,
            seed_val,
            gr.update(interactive=False, visible=False),
            None,
        )

    def on_image_confirmed(img):
        if img is not None:
            return (
                gr.update(interactive=True, visible=True),  # generate_3d_btn
                gr.update(visible=True),  # video_output
                gr.update(visible=True),  # model_output
                gr.update(interactive=False, visible=True),  # download_glb
                gr.update(interactive=False, visible=True),  # download_gs
            )
        return (
            gr.update(interactive=False, visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(interactive=False, visible=False),
            gr.update(interactive=False, visible=False),
        )

    generate_image_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        generate_text_to_image,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            enable_live_preview,
            use_quality_vae,
        ],
        outputs=[generated_image, image_seed, confirm_image_btn],
    ).then(
        on_image_generated,
        inputs=[generated_image, image_seed],
        outputs=[generated_image, image_seed, confirm_image_btn, current_image],
    )

    # Step 2: Generate 3D (background removal handled by TRELLIS)
    confirm_image_btn.click(
        on_image_confirmed,
        inputs=[current_image],
        outputs=[
            generate_3d_btn,
            video_output,
            model_output,
            download_glb,
            download_gs,
        ],
    )

    generate_3d_btn.click(
        get_seed,
        inputs=[randomize_seed_3d, seed_3d],
        outputs=[seed_3d],
    ).then(
        generate_3d_model,
        inputs=[
            current_image,
            seed_3d,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
            mesh_simplify,
            texture_size,
        ],
        outputs=[output_buf, video_output, model_output, download_glb, extract_gs_btn],
    ).then(
        lambda: gr.update(interactive=True, visible=True),
        outputs=[download_glb],
    )

    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.update(interactive=True, visible=True),
        outputs=[download_gs],
    )


if __name__ == "__main__":
    demo.launch()
