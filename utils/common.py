from contextlib import contextmanager
import gc
import time

import torch
import deepspeed.comm.comm as dist
import imageio
from safetensors import safe_open

import numpy as np
import torchvision
from einops import rearrange
import imageio


DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float8': torch.float8_e4m3fn}
VIDEO_EXTENSIONS = set(x.extension for x in imageio.config.video_extensions)
AUTOCAST_DTYPE = None


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
    imageio.mimsave(path, outputs, fps=fps)

def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None



def get_rank():
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@contextmanager
def zero_first():
    if not is_main_process():
        dist.barrier()
    yield
    if is_main_process():
        dist.barrier()


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def log_duration(name):
    start = time.time()
    try:
        yield
    finally:
        print(f'{name}: {time.time()-start:.3f}')


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_state_dict(path):
    if path.endswith('.safetensors'):
        return load_safetensors(path)
    else:
        return torch.load(path, weights_only=True)


def round_to_nearest_multiple(x, multiple):
    return int(round(x / multiple) * multiple)


def round_down_to_multiple(x, multiple):
    return int((x // multiple) * multiple)
