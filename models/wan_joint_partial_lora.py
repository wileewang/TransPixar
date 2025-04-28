from models.wan import *
import peft
from typing import Any, Optional, Union
from utils.common import cache_video

def add_modality_embedding(x, seq_lens, modality_embedding):
    modality_embedding = modality_embedding.to(x.device, dtype=x.dtype)

    if not isinstance(seq_lens, torch.Tensor):
        # Python scalar -> broadcast
        seq_lens = torch.full((x.shape[0],), seq_lens, dtype=torch.long, device=x.device)
    elif seq_lens.ndim == 0:
        # 0-dim tensor -> broadcast
        seq_lens = seq_lens.expand(x.shape[0])

    # Now seq_lens is always [b]
    for i in range(x.shape[0]):
        half = seq_lens[i] // 2
        x[i, half:seq_lens[i], :] += modality_embedding

    return x


def get_lora_output_mask(seq_len, start_idx, end_idx, scale_value=1.0, device='cuda'):
    """
    Returns a mask tensor of shape [1, seq_len, 1] that applies scale_value to positions [start_idx:end_idx],
    and 0 elsewhere. Suitable for direct multiplication with LoRA output.
    """
    mask = torch.zeros(seq_len, dtype=torch.float32, device=device)
    mask[start_idx:end_idx] = scale_value
    return mask.view(1, -1, 1)  # Shape: [1, seq_len, 1]



def register_model(model):
    def create_custom_model_forward(self):
        def custom_forward(
            x,
            t,
            context,
            seq_len,
            clip_fea=None,
            y=None,
        ):
            r"""
            Forward pass through the diffusion model

            Args:
                x (List[Tensor]):
                    List of input video tensors, each with shape [C_in, F, H, W]
                t (Tensor):
                    Diffusion timesteps tensor of shape [B]
                context (List[Tensor]):
                    List of text embeddings each with shape [L, C]
                seq_len (`int`):
                    Maximum sequence length for positional encoding
                clip_fea (Tensor, *optional*):
                    CLIP image features for image-to-video mode
                y (List[Tensor], *optional*):
                    Conditional video inputs for image-to-video mode, same shape as x

            Returns:
                List[Tensor]:
                    List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
            """
            if self.model_type == 'i2v':
                assert clip_fea is not None and y is not None
            # params
            device = self.patch_embedding.weight.device
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            # patch embeddings luozhou
            if hasattr(self, "patch_embedding_aux"):
                ret = []
                for u in x:
                    u_rgb, u_aux = u.chunk(2, dim=1)
                    # u_rgb = self.patch_embedding
                    ret.append(torch.cat([self.patch_embedding(u_rgb.unsqueeze(0)), self.patch_embedding_aux(u_aux.unsqueeze(0))], dim=2))
                x = ret
            else:
                x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                        dim=1) for u in x
            ])

            # time embeddings
            with amp.autocast(dtype=torch.float32):
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32

            # context
            context_lens = None
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))

            if clip_fea is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)

            
            # Luozhou
            if hasattr(self, "modality_embedding") and self.modality_embedding is not None:
                x = add_modality_embedding(x, seq_lens, self.modality_embedding)


            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens)

            for block in self.blocks:
                x = block(x, **kwargs)

            # head
            if hasattr(self, "head_aux"):
                x_rgb, x_aux = x.chunk(2, dim=1)
                x_rgb = self.head(x_rgb, e)
                x_aux = self.head_aux(x_aux, e)

                x = torch.cat([x_rgb, x_aux], dim=1)
            else:
                x = self.head(x, e)

            # unpatchify
            x = self.unpatchify(x, grid_sizes)
            return [u.float() for u in x]
        return custom_forward
    
    def create_custom_sa_forward(self):
        # self is captured from the outer scope, like model
        def custom_forward(x, seq_lens, grid_sizes, freqs):
            r"""
            Args:
                x (Tensor): Shape [B, L, num_heads, C / num_heads]
                seq_lens (Tensor): Shape [B]
                grid_sizes (Tensor): Shape [B, 3], the second dimension contains (F, H, W)
                freqs (Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            """
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

            def qkv_fn(x):
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)
                return q, k, v

            q, k, v = qkv_fn(x)

            x = flash_attention(
                q=rope_apply_control(q, grid_sizes, freqs),
                k=rope_apply_control(k, grid_sizes, freqs),  # Luozhou
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
            )

            x = x.flatten(2)
            x = self.o(x)
            return x

        return custom_forward


    model.forward = create_custom_model_forward(model)

    for _, module in model.named_modules():
        if type(module) is WanSelfAttention:
            module.forward = create_custom_sa_forward(module)


@amp.autocast(enabled=False)
def rope_apply_control(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        #! 
        # 这里应该不会有问题，因为RGB和Dep的frame数目是对齐的
        assert f % 2 == 0, f"Frame number f={f} is not even!"
        #!
        
        #!
        # 视作只有一半的长度
        # seq_len = f * h * w
        seq_len_half = (f // 2) * h * w
        #!

        #!
        # 分开为两部分，处理成RGB和Dep的token
        # precompute multipliers
        # x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
        #     seq_len, n, -1, 2))
        x_i_rgb = torch.view_as_complex(x[i, :seq_len_half].to(torch.float64).reshape(
            seq_len_half, n, -1, 2))
        x_i_dep = torch.view_as_complex(x[i, seq_len_half:(seq_len_half * 2)].to(torch.float64).reshape(
            seq_len_half, n, -1, 2))
        #!

        #!
        # 这部分代码就是RGB图像的每一帧每一个patch的位置信息
        # 此处需要注意因为帧数应该是只有原来的一半,所以这里的f都需要修改为f // 2
        freqs_i = torch.cat([
            # freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[0][:(f // 2)].view((f // 2), 1, 1, -1).expand((f // 2), h, w, -1),
            # freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand((f // 2), h, w, -1),
            # freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            freqs[2][:w].view(1, 1, w, -1).expand((f // 2), h, w, -1)
        ],
                            dim=-1).reshape(seq_len_half, 1, -1)
        #!

        #!
        # apply rotary embedding
        # 同样也是在rgb和dep上分别进行进行PE操作，位置相同，所以计算所用的freqs_i一致
        # x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i_rgb = torch.view_as_real(x_i_rgb * freqs_i).flatten(2)
        x_i_dep = torch.view_as_real(x_i_dep * freqs_i).flatten(2)
        # x_i = torch.cat([x_i, x[i, seq_len:]])
        x_i = torch.cat([x_i_rgb, x_i_dep, x[i, (seq_len_half * 2):]])
        #!

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanJointPipeline(WanPipeline):
    name = 'wan-joint'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['WanAttentionBlock']

    def __init__(self, config, dataset_config=None):
        super().__init__(config)
        width, height = dataset_config['resolutions'][0]
        frames = dataset_config['frame_buckets'][-1]

        # TXT_SEQ_LEN = 256
        IMG_SEQ_LEN = 2 * (width // 16) * (height // 16) * (1 + (frames - 1) // 4 )

        self.module_name_to_lora_modifier = {
            'self_attn': get_lora_output_mask(IMG_SEQ_LEN, start_idx=IMG_SEQ_LEN//2, end_idx=IMG_SEQ_LEN),
            'cross_attn.q': get_lora_output_mask(IMG_SEQ_LEN, start_idx=IMG_SEQ_LEN//2, end_idx=IMG_SEQ_LEN),
            'ffn': get_lora_output_mask(IMG_SEQ_LEN, start_idx=IMG_SEQ_LEN//2, end_idx=IMG_SEQ_LEN),
        }


    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if transformer_path := self.model_config.get('transformer_path', None):
            self.transformer = WanModelFromSafetensors.from_pretrained(
                transformer_path,
                self.original_model_config_path,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            self.transformer = WanModel.from_pretrained(self.model_config['ckpt_path'], torch_dtype=dtype)
            for name, p in self.transformer.named_parameters():
                if not (any(x in name for x in KEEP_IN_HIGH_PRECISION)):
                    p.data = p.data.to(transformer_dtype)

        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name
        
        # Luozhou prepare the modality embedding
        self.transformer.modality_embedding = torch.nn.Parameter(torch.zeros(1, self.transformer.config['dim'], device="cuda", dtype=dtype))

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        # add modality_embedding
        peft_state_dict['transformer.modality_embedding'] = self.transformer.modality_embedding.detach().cpu()

        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        aux_latents = inputs['aux_latents'].float()
        # TODO: why does text_embeddings become float32 here? It's bfloat16 coming out of the text encoder.
        text_embeddings = inputs['text_embeddings']
        seq_lens = inputs['seq_lens']
        mask = None # inputs['mask']
        y = inputs['y'] if self.i2v else None
        clip_context = inputs['clip_context'] if self.i2v else None

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)

        x_1 = torch.cat([latents, aux_latents], dim=2) #Luozhou
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000

        return (
            x_t,
            y,
            t,
            text_embeddings,
            seq_lens,
            clip_context,
            target
        )

    def to_layers(self):
        transformer = self.transformer
        layers = [JointInitialLayer(transformer)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    @torch.no_grad()
    def sample(self,
                 prompt=None,
                 context=None,
                 context_lens=None,
                 size=(832, 480),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1):
        F = frame_num
        device = 'cuda'
        num_train_timesteps = 1000
        target_shape = (self.vae.model.z_dim, 2 * ((F - 1) // self.vae_stride[0] + 1),
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        # Luozhou

        if n_prompt == "":
            n_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)

        self.transformer.to(device)
        self.transformer.eval()
        self.vae.model.to(device)
        self.vae.mean = self.vae.mean.to(device)
        self.vae.std = self.vae.std.to(device)
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        
        if prompt is not None:
            context = self.text_encoder([prompt], torch.device('cpu'))
            context = [t.to(device) for t in context]
        else:
            context = [emb[:length].to(device) for emb, length in zip(context, context_lens)]
        
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context_null = [t.to(device) for t in context_null]
        # else:
        #     context = self.text_encoder([input_prompt], torch.device('cpu'))
        #     context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        #     context = [t.to(self.device) for t in context]
        #     context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.transformer, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.model_config['dtype']), torch.no_grad(), no_sync():
        #with amp.autocast(dtype=self.model_config['dtype']), torch.no_grad():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)
                
                noise_pred_cond = self.transformer(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.transformer(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            # if offload_model:
            #     self.transformer.cpu()
            #     torch.cuda.empty_cache()
            # if self.rank == 0:

            # Assuming tensor_list is your list of tensors
            x0_rgb, x0_aux = zip(*[torch.chunk(t, chunks=2, dim=1) for t in x0])
            # Convert the result to lists if needed
            x0_rgb = list(x0_rgb)
            x0_aux = list(x0_aux)

            videos_rgb = self.vae.decode(x0_rgb)
            videos_aux = self.vae.decode(x0_aux)

        # del noise, latents
        # del sample_scheduler
        # if offload_model:
        #     gc.collect()
        #     torch.cuda.synchronize()
        # if dist.is_initialized():
        #     dist.barrier()

        return videos_rgb[0], videos_aux[0]

    def register_custom_op(self):

        # custom pe and learnable query at the beginning
        register_model(self.transformer)

        # partial lora
        def create_custom_lora_forward(self, lora_magnitude_vector):

            # based on lora forward of peft
            def custom_forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)
                adapter_names = kwargs.pop("adapter_names", None)

                if self.disable_adapters:
                    if self.merged:
                        self.unmerge()
                    result = self.base_layer(x, *args, **kwargs)
                elif adapter_names is not None:
                    result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
                elif self.merged:
                    result = self.base_layer(x, *args, **kwargs)
                else:
                    result = self.base_layer(x, *args, **kwargs)
                    torch_result_dtype = result.dtype
                    for active_adapter in self.active_adapters:
                        if active_adapter not in self.lora_A.keys():
                            continue
                        lora_A = self.lora_A[active_adapter]
                        lora_B = self.lora_B[active_adapter]
                        dropout = self.lora_dropout[active_adapter]
                        scaling = self.scaling[active_adapter]
                        x = x.to(lora_A.weight.dtype)

                        if not self.use_dora[active_adapter]:
                            result = result + lora_B(lora_A(dropout(x))) * scaling * lora_magnitude_vector.to(device=result.device, dtype=result.dtype)
                        else:
                            if isinstance(dropout, nn.Identity) or not self.training:
                                base_result = result
                            else:
                                x = dropout(x)
                                base_result = None

                            result = result + self.lora_magnitude_vector[active_adapter](
                                x,
                                lora_A=lora_A,
                                lora_B=lora_B,
                                scaling=scaling,
                                base_layer=self.get_base_layer(),
                                base_result=base_result,
                            )

                    result = result.to(torch_result_dtype)

                return result
            return custom_forward

        for name, module in self.transformer.named_modules():
            if isinstance(module, peft.tuners.lora.LoraLayer):
                for key, lora_magnitude_vector in self.module_name_to_lora_modifier.items():
                    if key in name:
                        print(f'Registering index modifier for LoRA module: {name}')
                        module.forward = create_custom_lora_forward(module, lora_magnitude_vector)
                        break

    def evaluate(self, data, save_path):
        text_embeddings = data[0][3]
        text_seq_lens = data[0][4]

        F = data[0][0].shape[2] // 2
        frame_num = 4 * (F - 1) + 1

        video_rgb, video_aux = self.sample(
            context=text_embeddings,
            context_lens=text_seq_lens,
            size=(832, 480),
            frame_num=frame_num,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=30,
            guide_scale=5.0,
        )


        save_path_rgb = save_path
        save_path_aux = save_path.replace('.mp4', '_aux.mp4')

        cache_video(
            tensor=video_rgb[None],
            save_file=save_path_rgb,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

        cache_video(
            tensor=video_aux[None],
            save_file=save_path_aux,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    
class JointInitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.freqs = model.freqs
        self.freq_dim = model.freq_dim
        self.dim = model.dim
        self.text_len = model.text_len


        self.i2v = (model.model_type == 'i2v')
        if self.i2v:
            self.img_emb = model.img_emb
        self.model = [model]
        self.model[0].modality_embedding.requires_grad_(True)
        self.register_parameter('modality_embedding', self.model[0].modality_embedding) # Luozhou

    # def __getattr__(self, name):
    #     return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, y, t, context, text_seq_lens, clip_fea, target = inputs
        bs, channels, f, h, w = x.shape
        if clip_fea.numel() == 0:
            clip_fea = None
        context = [emb[:length] for emb, length in zip(context, text_seq_lens)]

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        

        if self.i2v:
            mask = torch.zeros((bs, 4, f, h, w), device=x.device, dtype=x.dtype)
            mask[:, :, 0, ...] = 1
            y = torch.cat([mask, y], dim=1)
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.device, torch.float32))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.i2v:
            assert clip_fea is not None
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # pipeline parallelism needs everything on the GPU
        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)

        x = add_modality_embedding(x, seq_lens, self.modality_embedding)
        return make_contiguous(x, e, e0, seq_lens, grid_sizes, self.freqs, context, target)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, target = inputs 
        x = self.head(x, e) 
        x = self.unpatchify(x, grid_sizes) # [b,c,f,h,w]
        output = torch.stack(x, dim=0)
        with torch.autocast('cuda', enabled=False):
            output = output.to(torch.float32)
            target = target.to(torch.float32)
            loss = F.mse_loss(output, target)
        return loss