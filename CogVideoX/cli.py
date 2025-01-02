import os
import torch
from diffusers import CogVideoXDPMScheduler
from pipeline_rgba import CogVideoXPipeline
from diffusers.utils import export_to_video
import argparse
import numpy as np
from rgba_utils import *

def main(args):
    # 1. load pipeline  
    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


    # 2. define prompt and arguments
    pipeline_args = {
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "output_type": "latent",
        "use_dynamic_cfg":True,
    }

    # 3. prepare rgbx utils
    # breakpoint()
    seq_length = 2 * (
        (args.height // pipe.vae_scale_factor_spatial // 2)
        * (args.width // pipe.vae_scale_factor_spatial // 2)
        * ((args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1)
    )
    # seq_length = 35100

    prepare_for_rgba_inference(
        pipe.transformer,
        rgba_weights_path=args.lora_path,
        device="cuda",
        dtype=torch.bfloat16,
        text_length=226,
        seq_length=seq_length, # this is for the creation of attention mask.
    )

    # 4. run inference
    generator = torch.manual_seed(args.seed) if args.seed else None
    frames_latents = pipe(**pipeline_args, generator=generator).frames

    frames_latents_rgb, frames_latents_alpha = frames_latents.chunk(2, dim=1)

    frames_rgb = decode_latents(pipe, frames_latents_rgb)
    frames_alpha = decode_latents(pipe, frames_latents_alpha)


    pooled_alpha = np.max(frames_alpha, axis=-1, keepdims=True)
    frames_alpha_pooled = np.repeat(pooled_alpha, 3, axis=-1)
    premultiplied_rgb = frames_rgb * frames_alpha_pooled

    if os.path.exists(args.output_path) == False:
        os.makedirs(args.output_path)

    export_to_video(premultiplied_rgb[0], os.path.join(args.output_path, "rgb.mp4"), fps=args.fps)
    export_to_video(frames_alpha_pooled[0], os.path.join(args.output_path, "alpha.mp4"), fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument("--lora_path", type=str, default="/hpc2hdd/home/lwang592/projects/CogVideo/sat/outputs/training/ckpts-5b-attn_rebias-partial_lora-8gpu-wo_t2a/lora-rgba-12-21-19-11/5000/rgba_lora.safetensors", help="The path of the LoRA weights to be used")
    
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5B", help="Path of the pre-trained model use"
    )


    parser.add_argument("--output_path", type=str, default="./output", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=720, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=480, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=8, help="Number of steps for the inference process")
    parser.add_argument("--seed", type=int, default=None, help="The seed for reproducibility")
    args = parser.parse_args()

    main(args)