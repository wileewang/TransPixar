# TransPixeler

This project extends [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe), a pipeline-parallel training framework for diffusion models, to support a **joint generation** task involving paired video inputs and textual captions.

Key modifications include:

- Support for paired video inputs with fixed `_seg` suffix (e.g., `001.mp4` and `001_seg.mp4`);
- Customized dataset loading and preprocessing tailored for joint generation;
- Periodic evaluation and video sampling during training, configurable via `eval_every_step` or `eval_every_epoch`.

Currently, the implementation is available on the `wan` branch.



## Installing

Clone the repository and checkout the `wan` branch:

```bash
git clone --recurse-submodules -b wan https://github.com/wileewang/TransPixar.git
cd TransPixar
```

If you already cloned it and forgot to do `--recurse-submodules`:

```bash
git submodule init
git submodule update
```

Install Miniconda:  
👉 [Download Miniconda](https://docs.anaconda.com/miniconda/)

Create the environment:

```bash
conda create -n transpixeler python=3.10
conda activate transpixeler
```

Install `nvcc` (optional, if you plan to compile custom CUDA kernels):  
👉 [Install from Anaconda](https://anaconda.org/nvidia/cuda-nvcc)  
Try to match the CUDA version installed with your PyTorch.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```


## Dataset preparation

A dataset consists of one or more directories containing paired video files and corresponding captions.  
Each sample must include:

- A primary video file (e.g., `001.mp4`) representing the RGB content.
- A paired secondary video file (e.g., `001_seg.mp4`) with a fixed `_seg` suffix, representing the associated modality (e.g., segmentation, alpha mask).
- A caption text file (e.g., `001.txt`) with the same base name as the primary video.

**Important notes:**

- The `_seg` suffix in the secondary video file name is required and fixed.
- All three files (`.mp4`, `_seg.mp4`, `.txt`) must exist in the same directory for each sample.

An example of a valid data structure:

```
dataset/
  ├── 001.mp4
  ├── 001_seg.mp4
  ├── 001.txt
  ├── 002.mp4
  ├── 002_seg.mp4
  ├── 002.txt
  └── ...
```


## Training
**Start by reading through the config files in the examples directory.** Almost everything is commented, explaining what each setting does.

Once you've familiarized yourself with the config file format, go ahead and make a copy and edit to your liking. At minimum, change all the paths to conform to your setup, including the paths in the dataset config file.

Launch training like this:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config configs/wan.toml
```
RTX 4000 series needs those 2 environment variables set. Other GPUs may not need them. You can try without them, Deepspeed will complain if it's wrong.

If you enabled checkpointing, you can resume training from the latest checkpoint by simply re-running the exact same command but with the ```--resume_from_checkpoint``` flag.

## Output files
A new directory will be created in output_dir for each training run. This contains the checkpoints, saved models, and Tensorboard metrics. Saved models/LoRAs will be in directories named like epoch1, epoch2, etc. Deepspeed checkpoints are in directories named like global_step1234. These checkpoints contain all training state, including weights, optimizer, and dataloader state, but can't be used directly for inference. The saved model directory will have the safetensors weights, PEFT adapter config JSON, as well as the diffusion-pipe config file for easier tracking of training run settings.

If eval_every_step or eval_every_epoch is set in the config, the code will periodically sample generated videos during training and save them for evaluation.

## Parallelism
This code uses hybrid data- and pipeline-parallelism. Set the ```--num_gpus``` flag appropriately for your setup. Set ```pipeline_stages``` in the config file to control the degree of pipeline parallelism. Then the data parallelism degree will automatically be set to use all GPUs (number of GPUs must be divisible by pipeline_stages). For example, with 4 GPUs and pipeline_stages=2, you will run two instances of the model, each divided across two GPUs.

## Pre-caching
Latents and text embeddings are cached to disk before training happens. This way, the VAE and text encoders don't need to be kept loaded during training. The Huggingface Datasets library is used for all the caching. Cache files are reused between training runs if they exist. All cache files are written into a directory named "cache" inside each dataset directory.

This caching also means that training LoRAs for text encoders is not currently supported.

Two flags are relevant for caching. ```--cache_only``` does the caching flow, then exits without training anything. ```--regenerate_cache``` forces cache regeneration. If you edit the dataset in-place (like changing a caption), you need to force regenerate the cache (or delete the cache dir) for the changes to be picked up.

<!-- ## Extra
You can check out my [qlora-pipe](https://github.com/tdrussell/qlora-pipe) project, which is basically the same thing as this but for LLMs. -->
