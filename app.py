"""
THis is the main file for the gradio web demo. It uses the CogVideoX-5B model to generate videos gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.
Usage:
    OpenAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

import math
import os
import random
import threading
import time

import cv2
import tempfile
import imageio_ffmpeg
import gradio as gr
import torch
from PIL import Image
# from diffusers import (
#     CogVideoXPipeline,
#     CogVideoXDPMScheduler,
#     CogVideoXVideoToVideoPipeline,
#     CogVideoXImageToVideoPipeline,
#     CogVideoXTransformer3DModel,
# )
from typing import Union, List
from CogVideoX.pipeline_rgba import CogVideoXPipeline
from CogVideoX.rgba_utils import *
from diffusers import CogVideoXDPMScheduler

from diffusers.utils import load_video, load_image, export_to_video
from datetime import datetime, timedelta

from diffusers.image_processor import VaeImageProcessor
import moviepy.editor as mp
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

# hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="model_real_esran")
hf_hub_download(repo_id="wileewang/TransPixar", filename="cogvideox_rgba_lora.safetensors", local_dir="model_cogvideox_rgba_lora")
# snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5B", torch_dtype=torch.bfloat16)
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
seq_length = 2 * (
    (480 // pipe.vae_scale_factor_spatial // 2)
    * (720 // pipe.vae_scale_factor_spatial // 2)
    * ((13 - 1) // pipe.vae_scale_factor_temporal + 1)
)
prepare_for_rgba_inference(
    pipe.transformer,
    rgba_weights_path="model_cogvideox_rgba_lora/cogvideox_rgba_lora.safetensors",
    device="cuda",
    dtype=torch.bfloat16,
    text_length=226,
    seq_length=seq_length, # this is for the creation of attention mask.
)

# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
# pipe_image.transformer.to(memory_format=torch.channels_last)
# pipe_image.transformer = torch.compile(pipe_image.transformer, mode="max-autotune", fullgraph=True)

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

# upscale_model = utils.load_sd_upscale("model_real_esran/RealESRGAN_x4.pth", device)
# frame_interpolation_model = load_rife_model("model_rife")


sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.
For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:
You will only ever output a single video description per user request.
When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.
Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""
def save_video(tensor: Union[List[np.ndarray], List[Image.Image]], fps: int = 8, prefix='rgb'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{prefix}_{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=fps)
    return video_path

def resize_if_unfit(input_video, progress=gr.Progress(track_tqdm=True)):
    width, height = get_video_dimensions(input_video)

    if width == 720 and height == 480:
        processed_video = input_video
    else:
        processed_video = center_crop_resize(input_video)
    return processed_video


def get_video_dimensions(input_video_path):
    reader = imageio_ffmpeg.read_frames(input_video_path)
    metadata = next(reader)
    return metadata["size"]


def center_crop_resize(input_video_path, target_width=720, target_height=480):
    cap = cv2.VideoCapture(input_video_path)

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width_factor = target_width / orig_width
    height_factor = target_height / orig_height
    resize_factor = max(width_factor, height_factor)

    inter_width = int(orig_width * resize_factor)
    inter_height = int(orig_height * resize_factor)

    target_fps = 8
    ideal_skip = max(0, math.ceil(orig_fps / target_fps) - 1)
    skip = min(5, ideal_skip)  # Cap at 5

    while (total_frames / (skip + 1)) < 49 and skip > 0:
        skip -= 1

    processed_frames = []
    frame_count = 0
    total_read = 0

    while frame_count < 49 and total_read < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if total_read % (skip + 1) == 0:
            resized = cv2.resize(frame, (inter_width, inter_height), interpolation=cv2.INTER_AREA)

            start_x = (inter_width - target_width) // 2
            start_y = (inter_height - target_height) // 2
            cropped = resized[start_y : start_y + target_height, start_x : start_x + target_width]

            processed_frames.append(cropped)
            frame_count += 1

        total_read += 1

    cap.release()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_video_path = temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (target_width, target_height))

        for frame in processed_frames:
            out.write(frame)

        out.release()

    return temp_video_path



def infer(
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = -1,
    progress=gr.Progress(track_tqdm=True),
):
    if seed == -1:
        seed = random.randint(0, 2**8 - 1)
    pipe.to(device)
    video_pt = pipe(
        prompt=prompt + ", isolated background",
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=13,
        use_dynamic_cfg=True,
        output_type="latent",
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(int(seed)),
    ).frames
    # pipe.to("cpu")
    gc.collect()
    return (video_pt, seed)


def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.HTML("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               TransPixar + CogVideoX-5B Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/wileewang/TransPixar">🤗 TransPixar LoRA Hub</a> |
               <a href="https://github.com/wileewang/TransPixar">🌐 Github</a> |
               <a href="https://arxiv.org/">📜 arxiv </a>
           </div>
           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. 
            </div>
           """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=-1
                        )

            generate_button = gr.Button("🎬 Generate Video")
            with gr.Row():
                gr.Markdown(
                    """
                    **Note:** The output RGB is a premultiplied version to avoid the color decontamination problem.
                    It can directly composite with a background using:
                    ```
                    composite = rgb + (1 - alpha) * background
                    ```
                    """
                )

        with gr.Column():
            rgb_video_output = gr.Video(label="Generated RGB Video", width=720, height=480)
            alpha_video_output = gr.Video(label="Generated Alpha Video", width=720, height=480)
            with gr.Row():
                download_rgb_video_button = gr.File(label="📥 Download RGB Video", visible=False)
                download_alpha_video_button = gr.File(label="📥 Download Alpha Video", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)


    def generate(
        prompt,
        seed_value,
        progress=gr.Progress(track_tqdm=True)
    ):
        latents, seed = infer(
            prompt,
            num_inference_steps=25,  # NOT Changed
            guidance_scale=7.0,  # NOT Changed
            seed=seed_value,
            progress=progress,
        )

        latents_rgb, latents_alpha = latents.chunk(2, dim=1)

        frames_rgb = decode_latents(pipe, latents_rgb)
        frames_alpha = decode_latents(pipe, latents_alpha)

        pooled_alpha = np.max(frames_alpha, axis=-1, keepdims=True)
        frames_alpha_pooled = np.repeat(pooled_alpha, 3, axis=-1)
        premultiplied_rgb = frames_rgb * frames_alpha_pooled

        rgb_video_path = save_video(premultiplied_rgb[0], fps=8, prefix='rgb')
        rgb_video_update = gr.update(visible=True, value=rgb_video_path)

        alpha_video_path = save_video(frames_alpha_pooled[0], fps=8, prefix='alpha')
        alpha_video_update = gr.update(visible=True, value=alpha_video_path)
        seed_update = gr.update(visible=True, value=seed)

        return rgb_video_path, alpha_video_path, rgb_video_update, alpha_video_update, seed_update


    generate_button.click(
        generate,
        inputs=[prompt, seed_param],
        outputs=[rgb_video_output, alpha_video_output, download_rgb_video_button, download_alpha_video_button, seed_text],
    )


if __name__ == "__main__":
    demo.queue(max_size=15)
    demo.launch()
