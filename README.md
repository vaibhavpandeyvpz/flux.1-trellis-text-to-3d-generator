---
title: Flux.1 Trellis Text to 3D Generator
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Generate 3D models from text using Flux.1 + rembg + TRELLIS
---

# 🎨 Text to Image to 3D Generator

Complete end-to-end workflow for generating 3D models from text prompts using FLUX.1 and TRELLIS models.

## 🚀 Features

- **Step 1: Text to Image**: Generate high-quality images from text prompts using FLUX.1 [dev]
- **Step 2: Background Removal**: Automatically remove backgrounds using rembg
- **Step 3: 3D Generation**: Convert images to 3D models using Microsoft's TRELLIS
- **Interactive Workflow**: Each step requires user confirmation before proceeding
- **Multiple Output Formats**: GLB files and Gaussian Splatting (.ply) files
- **Live Preview**: Watch images generate in real-time during text-to-image generation

## 📖 How to Use

### Step 1: Text to Image
1. Enter your text prompt in the input field
2. Adjust settings (optional):
   - Image dimensions (width/height)
   - Guidance scale and inference steps
   - Enable live preview for real-time generation
   - Choose between fast or quality VAE
3. Click "Generate Image"
4. Review the generated image
5. Click "Confirm & Remove Background" to proceed

### Step 2: Background Removal
1. Review the image with background
2. Click "Remove Background" to process
3. Review the image with transparent background (RGBA)
4. Click "Confirm & Generate 3D" to proceed

### Step 3: 3D Model Generation
1. Adjust 3D generation settings (optional):
   - Sparse Structure Generation parameters (Stage 1)
   - Structured Latent Generation parameters (Stage 2)
   - Mesh simplification and texture size
2. Click "Generate 3D Model"
3. Wait for generation to complete (typically 2-5 minutes)
4. Preview the 3D model in the interactive viewer
5. Download the GLB file or extract Gaussian splatting data

## 🔧 Setup

### Authentication

This app requires Hugging Face authentication for FLUX.1 model access:

1. Go to [FLUX.1-dev model page](https://huggingface.co/black-forest-labs/FLUX.1-dev) and accept the license
2. Create a Hugging Face token at https://huggingface.co/settings/tokens
3. In your Hugging Face Space settings, add a secret named `HF_TOKEN` with your token value

The code will automatically use the token from the environment variable to authenticate when loading the model.

### Required Files

This app requires the `trellis` module to be present in the root directory. The trellis module should be copied from the reference implementation or obtained from the TRELLIS repository.

## 💡 Tips for Best Results

### Text to Image
- Be descriptive and specific in your prompts
- Use quality VAE for final output if you have time
- Enable live preview to see progress

### Background Removal
- Images with clear subjects work best
- The tool automatically handles background removal

### 3D Generation
- Use clear, well-lit images with good contrast
- Images with transparent backgrounds (alpha channel) work best
- Ensure the main object is clearly visible and centered
- Higher guidance strength and more sampling steps may improve quality but take longer

## 🔧 Technical Details

### Models Used
- **FLUX.1 [dev]**: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
  - 12B parameter rectified flow transformer
  - Non-commercial license
- **TRELLIS**: [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large)
  - Scalable and versatile 3D generation model
- **rembg**: Automatic background removal

### Hardware
- **ZeroGPU (T4)**: GPU resources are allocated on-demand
- Processing time varies based on GPU availability and image complexity

### Output Formats
- **GLB**: Universal 3D format compatible with most 3D software, game engines, and web viewers
- **Gaussian Splatting (.ply)**: Advanced point-based representation for high-quality rendering (~50MB files)

## 📚 Resources

- [FLUX.1 Blog](https://blackforestlabs.ai/announcing-black-forest-labs/)
- [TRELLIS Project Page](https://trellis3d.github.io/)
- [TRELLIS Paper](https://huggingface.co/papers/2412.01506)

## ⚠️ Notes

- Processing requires GPU resources - you may need to wait if all GPUs are in use
- Gaussian splatting files can be large (~50MB) and may take time to download
- Each step in the workflow requires user confirmation before proceeding
- The app uses ZeroGPU, so GPU resources are allocated on-demand

## 📝 License

- FLUX.1 [dev]: Non-commercial license - see [LICENSE](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
- TRELLIS: MIT License
- This Space: Apache 2.0

---

Built with [Gradio](https://gradio.app/) and powered by [Hugging Face Spaces](https://huggingface.co/spaces)
