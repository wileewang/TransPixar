import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time
import random
import json
import inspect

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, save_videos_grid, cache_video
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches

#TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TIMESTEP_QUANTILES_FOR_EVAL = [0.1]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--dataset_config', help='Path to TOML configuration file.')
parser.add_argument('--ckpt', default=None, help='Path to CKPT file.')
parser.add_argument('--output_dir', default='/output', help='Path to output.')

parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None, help='resume training from the most recent checkpoint')
parser.add_argument('--regenerate_cache', action='store_true', default=None, help='Force regenerate cache. Useful if none of the files have changed but their contents have, e.g. modified captions.')
parser.add_argument('--cache_only', action='store_true', default=None, help='Cache model inputs then exit.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', default=None, help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    # assert 'save_every_n_epochs' in config

    # config.setdefault('pipeline_stages', 1)
    # config.setdefault('activation_checkpointing', False)
    # config.setdefault('warmup_steps', 0)
    # if 'save_dtype' in config:
    #     config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            # if 'alpha' in adapter_config:
            #     raise NotImplementedError(
            #         'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
            #     )
            # adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    # config.setdefault('logging_steps', 1)
    # config.setdefault('eval_datasets', [])
    # config.setdefault('eval_gradient_accumulation_steps', 1)
    # config.setdefault('eval_every_n_steps', None)
    # config.setdefault('eval_every_n_epochs', None)
    # config.setdefault('eval_before_first_step', True)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


def evaluate_single(model, eval_dataloader, eval_gradient_accumulation_steps, quantile, step, epoch, eval_dir, model_type):
    eval_dataloader.set_eval_quantile(quantile)
    iterator = iter(eval_dataloader)
    num = 0

    # eval_dir_current = os.path.join(eval_dir, f'epoch-{epoch}_step-{step}')
    # os.makedirs(eval_dir_current, exist_ok=True)
    samples_per_gpu = len(eval_dataloader)
    if is_main_process():
        print(f'evaluating for epoch-{epoch}_step-{step}......')

    while True:
        data = iterator.__next__()
        if "wan" in model_type:
            save_path = os.path.join(eval_dir, f'sample-{get_rank()*samples_per_gpu+num}.mp4')
            model.evaluate(data, save_path)
        if "hunyuan" in model_type:
            prompt_embeds_1, prompt_attention_mask_1, prompt_embeds_2 = data[0][2], data[0][3], data[0][4]
        # if 'wan' in model_type:
        #     text_embeddings = data[0][3]
        #     text_seq_lens = data[0][4]

        # (
        #     x_t,
        #     y,
        #     t,
        #     text_embeddings,
        #     seq_lens,
        #     clip_context,
        #     target
        # )
        # device = "cuda"
        # model.vae.model.to(device)
        # model.vae.mean = model.vae.mean.to(device)
        # model.vae.std = model.vae.std.to(device)
        # model.vae.scale = [model.vae.mean, 1.0 / model.vae.std]

        # t = data[0][2][[0,]] / 1000
        # t_expanded = t.view(-1, 1, 1, 1, 1)

        # x_t = data[0][0][[0,]] # (1 - t_expanded) * x_1 + t_expanded * x_0


        # target = data[0][-1][[0,]] # x_0 - x_1

        # x_1 = x_t - t_expanded * target

        # latents_rgb, latents_aux = x_1.chunk(2, dim=2)

        # latents_rgb = latents_rgb.to(dtype=model.model_config['dtype'], device="cuda")
        # latents_aux = latents_aux.to(dtype=model.model_config['dtype'], device="cuda")

        # video_rgb = model.vae.decode(latents_rgb)
        # video_aux = model.vae.decode(latents_aux)
        
        # save_path_rgb = f'/data/user/user69/projects/video-generation-runs/test_dataloader_lz_8gpu_train/rgb_{dist.get_rank()}_{num}.mp4'
        # save_path_aux = f'/data/user/user69/projects/video-generation-runs/test_dataloader_lz_8gpu_train/aux_{dist.get_rank()}_{num}.mp4'

        # save_videos_grid(video_rgb[0].cpu()[None], save_path_rgb, fps=24)
        # save_videos_grid(video_aux[0].cpu()[None], save_path_aux, fps=24)

        # # cache_video(
        # #     tensor=video_rgb[None],
        # #     save_file=save_path_rgb,
        # #     fps=30,
        # #     nrow=1,
        # #     normalize=True,
        # #     value_range=(-1, 1))

        # # cache_video(
        # #     tensor=video_aux[None],
        # #     save_file=save_path_aux,
        # #     fps=30,
        # #     nrow=1,
        # #     normalize=True,
        # #     value_range=(-1, 1))



        ### insert sample function here
        if model_type =='hunyuan-video':
            sample = model.sample(prompt=None, prompt_embeds_1=prompt_embeds_1, prompt_attention_mask_1=prompt_attention_mask_1, prompt_embeds_2=prompt_embeds_2, infer_steps=30)
            save_path = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
            save_videos_grid(sample[0][0].unsqueeze(0), save_path, fps=24)
        
        elif model_type =='hunyuan-video-joint' or model_type=='hunyuan-video-joint-partial-lora':
            sample = model.sample(
                prompt=None, 
                prompt_embeds_1=prompt_embeds_1, 
                prompt_attention_mask_1=prompt_attention_mask_1, 
                prompt_embeds_2=prompt_embeds_2, 
                infer_steps=30, 
                video_length=49
            ) # Luozhou
            save_path_rgb = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
            save_path_aux = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}_aux.mp4')
            save_videos_grid(sample[0][0].unsqueeze(0), save_path_rgb, fps=24)
            save_videos_grid(sample[1][0].unsqueeze(0), save_path_aux, fps=24)
        
        elif model_type =='hunyuan-video-i2v':
            # t = data[0][1][[0,]] / 1000
            # t_expanded = t.view(-1, 1, 1, 1, 1)

            x_t = data[0][0][[0,]] # (1 - t_expanded) * x_1 + t_expanded * x_0

            # target = data[0][-1][[0,]] # x_0 - x_1

            # x_1 = x_t - t_expanded * target

            sample = model.sample(image=x_t[:,:,[0,],:,:].to(dtype=model.dtype, device="cuda"), prompt=None, prompt_embeds_1=prompt_embeds_1, prompt_attention_mask_1=prompt_attention_mask_1, prompt_embeds_2=prompt_embeds_2, infer_steps=30, video_length=85) # Luozhou
            save_path = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
            save_videos_grid(sample[0][0].unsqueeze(0), save_path, fps=24)

        elif model_type =='hunyuan-video-control':
            x_t = data[0][0][[0,]] # (1 - t_expanded) * x_1 + t_expanded * x_0
            x_t_latents, video_condition = torch.chunk(x_t, 2, 2)        
            sample = model.sample(video_condition.to(dtype=model.dtype, device="cuda"), prompt=None, prompt_embeds_1=prompt_embeds_1, prompt_attention_mask_1=prompt_attention_mask_1, prompt_embeds_2=prompt_embeds_2, infer_steps=30)
            save_path = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
            save_videos_grid(sample[0][0].unsqueeze(0), save_path, fps=24)

        # elif model_type =='wan':
        #     video = model.sample(
        #         # prompt='a man is smiling',
        #         context=text_embeddings,
        #         context_lens=text_seq_lens,
        #         size=(832, 480),
        #         frame_num=81,
        #         shift=5.0,
        #         sample_solver='unipc',
        #         sampling_steps=30,
        #         guide_scale=5.0,
        #     )
        #     save_path = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
        #     cache_video(
        #         tensor=video[None],
        #         save_file=save_path,
        #         fps=30,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))
        
        # elif model_type =='wan-mul-scene':
        #     video = model.sample(
        #         # prompt='a man is smiling',
        #         scene_label=data[0][-2][0,],
        #         context=text_embeddings,
        #         context_lens=text_seq_lens,
        #         size=(832, 480),
        #         frame_num=81,
        #         shift=5.0,
        #         sample_solver='unipc',
        #         sampling_steps=30,
        #         guide_scale=5.0,
        #     )
        #     save_path = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
        #     cache_video(
        #         tensor=video[None],
        #         save_file=save_path,
        #         fps=30,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))
        
        # elif 'wan-joint' in model_type:
        #     video_rgb, video_aux = model.sample(
        #         # prompt='a man is smiling',
        #         context=text_embeddings,
        #         context_lens=text_seq_lens,
        #         size=(832, 480),
        #         frame_num=33,
        #         shift=5.0,
        #         sample_solver='unipc',
        #         sampling_steps=30,
        #         guide_scale=5.0,
        #     )

        #     save_path_rgb = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}_rgb.mp4')
        #     save_path_aux = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}_aux.mp4')

        #     cache_video(
        #         tensor=video_rgb[None],
        #         save_file=save_path_rgb,
        #         fps=16,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))

        #     cache_video(
        #         tensor=video_aux[None],
        #         save_file=save_path_aux,
        #         fps=16,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))
            
        # elif 'wan-control' in model_type:
        #     x_t = data[0][0][[0,]] # (1 - t_expanded) * x_1 + t_expanded * x_0
        #     x_t_latents, video_condition = torch.chunk(x_t, 2, 2)
        #     video = model.sample(
        #         video_condition,
        #         context=text_embeddings,
        #         context_lens=text_seq_lens,
        #         shift=5.0,
        #         sample_solver='unipc',
        #         sampling_steps=30,
        #         guide_scale=5.0,
        #     )
        #     save_path = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
        #     cache_video(
        #         tensor=video[None],
        #         save_file=save_path,
        #         fps=16,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))
            
        #     cond = model.vae.decode(video_condition.cuda())[0]
        #     save_path_cond = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}_cond.mp4')
        #     cache_video(
        #         tensor=cond[None],
        #         save_file=save_path_cond,
        #         fps=16,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))
        
        # elif 'wan-i2v' in model_type:
        #     y, clip_context = data[0][1], data[0][5]
        #     f = y.shape[2]
        #     F = (f - 1) * 4 + 1
        #     video = model.sample(
        #         # prompt='a man is smiling',
        #         img_latent=y,
        #         context=text_embeddings,
        #         context_lens=text_seq_lens,
        #         clip_context=clip_context,
        #         max_area=480 * 832,
        #         frame_num=F,
        #         shift=5.0,
        #         sample_solver='unipc',
        #         sampling_steps=30,
        #         guide_scale=5.0,
        #     )
        #     save_path = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}.mp4')
        #     cache_video(
        #         tensor=video[None],
        #         save_file=save_path,
        #         fps=16,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))
        #     cond = model.vae.decode(y.cuda())[0]
        #     save_path_cond = os.path.join(eval_dir_current, f'epoch-{epoch}_step-{step}_sample-{get_rank()*samples_per_gpu+num}_cond.mp4')
        #     cache_video(
        #         tensor=cond[None],
        #         save_file=save_path_cond,
        #         fps=16,
        #         nrow=1,
        #         normalize=True,
        #         value_range=(-1, 1))


        num += 1
        #### insert sample function here
        eval_dataloader.sync_epoch()
        
        if eval_dataloader.epoch == 2:
            break
    eval_dataloader.reset()
    # wait for every process finish evaluation
    dist.barrier()
    #model_engine.micro_batches = orig_micro_batches
    return


def _evaluate(model, eval_dataloaders, tb_writer, eval_gradient_accumulation_steps, step, epoch, eval_dir, model_type):
    if is_main_process():
        print('Running eval')

    quantile = TIMESTEP_QUANTILES_FOR_EVAL[0]
    for name, eval_dataloader in eval_dataloaders.items():
        evaluate_single(model, eval_dataloader, eval_gradient_accumulation_steps, quantile, step, epoch, eval_dir, model_type)


def evaluate(model, eval_dataloaders, tb_writer, eval_gradient_accumulation_steps, step, epoch, eval_dir, model_type):
    if len(eval_dataloaders) == 0:
        return
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if "hunyuan" in model_type:
            # set transformer to eval state
            model.diffusers_pipeline.transformer.eval()
            model.diffusers_pipeline.vae.to('cuda')
        elif "wan" in model_type:
            model.transformer.eval()
            model.vae.model.to('cuda')
        _evaluate(model, eval_dataloaders, tb_writer, eval_gradient_accumulation_steps, step, epoch, eval_dir, model_type)
        # reset transformer to train state
        if "hunyuan" in model_type:
            # set transformer to eval state
            model.diffusers_pipeline.vae.to('cpu')
            model.diffusers_pipeline.transformer.train()
        elif "wan" in model_type:
            model.vae.model.to('cpu')
            model.transformer.train()

if __name__ == '__main__':
    apply_patches()

    # needed for broadcasting Queue in dataset.py
    mp.current_process().authkey = b'afsaskgfdjh4'

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']

    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    deepspeed.init_distributed()
    # needed for broadcasting Queue in dataset.py (because we haven't called deepspeed.initialize() yet?)
    torch.cuda.set_device(dist.get_rank())

    model_type = config['model']['type']

    with open(args.dataset_config) as f:
        dataset_config = toml.load(f)

    
    if model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'hunyuan-video-joint':
        from models import hunyuan_video_joint
        model = hunyuan_video_joint.HunyuanVideoJointPipeline(config)
    elif model_type == 'hunyuan-video-joint-partial-lora':
        from models import hunyuan_video_joint_partial_lora
        model = hunyuan_video_joint_partial_lora.HunyuanVideoJointPipeline(config, dataset_config)
    elif model_type == 'hunyuan-video-i2v':
        from models import hunyuan_video_i2v
        model = hunyuan_video_i2v.HunyuanVideoI2VPipeline(config)
    elif model_type == 'hunyuan-video-control':
        from models import hunyuan_video_control
        model = hunyuan_video_control.HunyuanVideoControlPipeline(config)
    elif model_type == 'wan':
        from models import wan
        model = wan.WanPipeline(config)
    elif model_type == 'wan-i2v':
        from models import wan_i2v
        model = wan_i2v.WanPipeline(config)
    elif model_type == 'wan-joint':
        from models import wan_joint
        model = wan_joint.WanJointPipeline(config)
    elif model_type == 'wan-control':
        from models import wan_control
        model = wan_control.WanControlPipeline(config)
    elif model_type == 'wan-control-merge':
        from models import wan_control_merge
        model = wan_control_merge.WanControlPipeline(config)
    elif model_type == 'wan-control-similar':
        from models import wan_control_similar
        model = wan_control_similar.WanControlPipeline(config)
    elif model_type == 'wan-joint-jam':
        from models import wan_joint_jam
        model = wan_joint_jam.WanJointPipeline(config)
    elif model_type == 'wan-joint-partial-lora':
        from models import wan_joint_partial_lora
        model = wan_joint_partial_lora.WanJointPipeline(config, dataset_config)
    elif model_type == 'wan-control-partial-lora':
        from models import wan_control_partial_lora
        model = wan_control_partial_lora.WanControlPipeline(config, dataset_config)
    elif model_type == 'wan-mul-scene':
        from models import wan_mul_scene
        model = wan_mul_scene.WanMultiScenePipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)


    eval_data_map = {}
    name = 'test'
    eval_data_map[name] = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(eval_data_map[name])

    # for i, eval_dataset in enumerate():
    #     if type(eval_dataset) == str:
    #         name = f'eval{i}'
    #         config_path = eval_dataset
    #     else:
    #         name = eval_dataset['name']
    #         config_path = eval_dataset['config']
    #     with open(config_path) as f:
    #         eval_dataset_config = toml.load(f)
    #     eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    #     dataset_manager.register(eval_data_map[name])

    dataset_manager.cache()
    if args.cache_only:
        quit()

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        if init_from_existing := args.ckpt:
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    dist.barrier()

    # create directory to save evaluation video results
    # eval_dir = os.path.join(config['output_dir'], 'evaluation')
    suffix = os.path.basename(args.ckpt)
    eval_dir = os.path.join(args.output_dir, suffix)
    os.makedirs(eval_dir, exist_ok=True)

    model.register_custom_op()

    for eval_data in eval_data_map.values():
        eval_data.post_init(
            dist.get_rank(),
            dist.get_world_size(),
            config.get('eval_micro_batch_size_per_gpu', 1),
            config['eval_gradient_accumulation_steps'],
        )

    eval_dataloaders = {
        # Set num_dataloader_workers=0 so dataset iteration is completely deterministic.
        # We want the exact same noise for each image, each time, for a stable validation loss.
        name: dataset_util.PipelineDataLoader(eval_data, config.get('eval_gradient_accumulation_steps', 1), model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    # epoch = train_dataloader.epoch
    tb_writer = None
    # saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    # we have to reload the vae to ensure vae on correct device 
    # guibao
    model.reload_vae()
    # model.reload_text_encoder()
    # if config['eval_before_first_step']:
    evaluate(model, eval_dataloaders, tb_writer, config.get('eval_gradient_accumulation_steps', 1), 0, 0, eval_dir, model_type)

    if is_main_process():
        print('INFERENCE COMPLETE!')
