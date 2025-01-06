import os
import torch
# from diffusers import MochiPipeline
from pipeline_mochi_rgba import MochiPipeline
from diffusers.utils import export_to_video
import argparse
from rgba_utils import *
import numpy as np


def main(args):
    # 1. load pipeline  
    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16).to("cuda")
    pipe.enable_vae_tiling()

    # 2. define prompt and arguments
    pipeline_args = {
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "max_sequence_length": 256,
        "output_type": "latent",
    }

    # 3. prepare rgbx utils    
    prepare_for_rgba_inference(
        pipe.transformer,
        device="cuda",
        dtype=torch.bfloat16,
    )

    if args.lora_path is not None:
        checkpoint = torch.load(args.lora_path, map_location="cpu")
        processor_state_dict = checkpoint["state_dict"]
        load_processor_state_dict(pipe.transformer, processor_state_dict)


    # 4. inference
    generator = torch.manual_seed(args.seed) if args.seed else None
    frames_latents = pipe(**pipeline_args, generator=generator).frames

    frames_latents_rgb, frames_latents_alpha = frames_latents.chunk(2, dim=2)
    
    frames_rgb = decode_latents(pipe, frames_latents_rgb)
    frames_alpha = decode_latents(pipe, frames_latents_alpha)

    pooled_alpha = np.max(frames_alpha, axis=-1, keepdims=True)
    frames_alpha_pooled = np.repeat(pooled_alpha, 3, axis=-1)
    premultiplied_rgb = frames_rgb * frames_alpha_pooled

    if os.path.exists(args.output_path) == False:
        os.makedirs(args.output_path)

    export_to_video(premultiplied_rgb[0], os.path.join(args.output_path, "rgb.mp4"), fps=args.fps)
    export_to_video(frames_alpha_pooled[0], os.path.join(args.output_path, "alpha.mp4"), fps=args.fps)
    export_to_video(frames_rgb[0], os.path.join(args.output_path, "original_rgb.mp4"), fps=args.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    
    parser.add_argument(
        "--model_path", type=str, default="genmo/mochi-1-preview", help="Path of the pre-trained model use"
    )
    parser.add_argument("--output_path", type=str, default="./output", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=64, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=79, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=848, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=480, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=30, help="Number of steps for the inference process")
    parser.add_argument("--seed", type=int, default=None, help="The seed for reproducibility")
    args = parser.parse_args()

    main(args)
