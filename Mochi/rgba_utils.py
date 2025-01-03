import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@torch.no_grad()
def decode_latents(pipe, latents):
    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / pipe.vae.config.scaling_factor

    video = pipe.vae.decode(latents, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type='np')

    return video


class RGBALoRAMochiAttnProcessor:
    """Attention processor used in Mochi."""

    def __init__(self, device, dtype, lora_rank=128, lora_alpha=1.0, latent_dim=3072):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MochiAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

        
        # Initialize LoRA layers
        self.lora_alpha = lora_alpha
        self.lora_rank = lora_rank

        # Helper function to create LoRA layers
        def create_lora_layer(in_dim, mid_dim, out_dim, device=device, dtype=dtype):
            # Define the LoRA layers
            lora_a = nn.Linear(in_dim, mid_dim, bias=False, device=device, dtype=dtype)
            lora_b = nn.Linear(mid_dim, out_dim, bias=False, device=device, dtype=dtype)
            
            # Initialize lora_a with random parameters (default initialization)
            nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))  # or another suitable initialization
            
            # Initialize lora_b with zero values
            nn.init.zeros_(lora_b.weight)

            lora_a.weight.requires_grad = True
            lora_b.weight.requires_grad = True
            
            # Combine the layers into a sequential module
            return nn.Sequential(lora_a, lora_b)

        self.to_q_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_k_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_v_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_out_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)

    
    def _apply_lora(self, hidden_states, seq_len, query, key, value, scaling):
        """Applies LoRA updates to query, key, and value tensors."""
        query_delta = self.to_q_lora(hidden_states).to(query.device)
        query[:, -seq_len // 2:, :] += query_delta[:, -seq_len // 2:, :] * scaling

        key_delta = self.to_k_lora(hidden_states).to(key.device)
        key[:, -seq_len // 2:, :] += key_delta[:, -seq_len // 2:, :] * scaling

        value_delta = self.to_v_lora(hidden_states).to(value.device)
        value[:, -seq_len // 2:, :] += value_delta[:, -seq_len // 2:, :] * scaling

        return query, key, value

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        scaling = self.lora_alpha / self.lora_rank
        sequence_length = query.size(1)
        query, key, value = self._apply_lora(hidden_states, sequence_length, query, key, value, scaling)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)

            query[:,sequence_length//2:] = apply_rotary_emb(query[:,sequence_length//2:], *image_rotary_emb)
            query[:,:sequence_length//2] = apply_rotary_emb(query[:,:sequence_length//2], *image_rotary_emb)

            key[:,sequence_length//2:] = apply_rotary_emb(key[:,sequence_length//2:], *image_rotary_emb)
            key[:,:sequence_length//2] = apply_rotary_emb(key[:,:sequence_length//2], *image_rotary_emb)
            # query = apply_rotary_emb(query, *image_rotary_emb)
            # key = apply_rotary_emb(key, *image_rotary_emb)
            

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        encoder_query, encoder_key, encoder_value = (
            encoder_query.transpose(1, 2),
            encoder_key.transpose(1, 2),
            encoder_value.transpose(1, 2),
        )

        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)
        total_length = sequence_length + encoder_sequence_length

        batch_size, heads, _, dim = query.shape
        
        attn_outputs = []
        prompt_attention_mask = attention_mask["prompt_attention_mask"]
        rect_attention_mask = attention_mask["rect_attention_mask"]
        for idx in range(batch_size):
            mask = prompt_attention_mask[idx][None, :] # two components: attention mask and prompt mask
            valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

            valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

            valid_query = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
            valid_key = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
            valid_value = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)

            attn_output = F.scaled_dot_product_attention(
                valid_query, 
                valid_key, 
                valid_value, 
                dropout_p=0.0,
                attn_mask=rect_attention_mask[idx], 
                is_causal=False
            )
            valid_sequence_length = attn_output.size(2)
            attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
            attn_outputs.append(attn_output)

        hidden_states = torch.cat(attn_outputs, dim=0)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        # linear proj
        original_hidden_states = attn.to_out[0](hidden_states)
        hidden_states_delta = self.to_out_lora(hidden_states).to(hidden_states.device)
        original_hidden_states[:, -sequence_length // 2:, :] += hidden_states_delta[:, -sequence_length // 2:, :] * scaling
        # dropout
        hidden_states = attn.to_out[1](original_hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

def prepare_for_rgba_inference(
    model, device: torch.device, dtype: torch.dtype,
    lora_rank: int = 128, lora_alpha: float = 1.0
):

    def custom_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_attention_mask: torch.Tensor,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ) -> torch.Tensor:
            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)
            else:
                if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                    logger.warning(
                        "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                    )

            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            p = self.config.patch_size

            post_patch_height = height // p
            post_patch_width = width // p

            temb, encoder_hidden_states = self.time_embed(
                timestep,
                encoder_hidden_states,
                encoder_attention_mask["prompt_attention_mask"],
                hidden_dtype=hidden_states.dtype,
            )

            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.patch_embed(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

            image_rotary_emb = self.rope(
                self.pos_frequencies,
                num_frames // 2, # Identitical PE for RGB and Alpha
                post_patch_height,
                post_patch_width,
                device=hidden_states.device,
                dtype=torch.float32,
            )

            for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        encoder_attention_mask,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                    )
            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
            hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
            output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)
        return forward
    
    for _, block in enumerate(model.transformer_blocks):
        attn_processor = RGBALoRAMochiAttnProcessor(
            device=device, 
            dtype=dtype,
            lora_rank=lora_rank, 
            lora_alpha=lora_alpha
        )
        # block.attn1.set_processor(attn_processor)
        block.attn1.processor = attn_processor

    model.forward = custom_forward(model)

def get_processor_state_dict(model):
    """Save trainable parameters of processors to a checkpoint."""
    processor_state_dict = {}

    for index, block in enumerate(model.transformer_blocks):
        if hasattr(block.attn1, "processor"):
            processor = block.attn1.processor
            for attr_name in ["to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"]:
                if hasattr(processor, attr_name):
                    lora_layer = getattr(processor, attr_name)
                    for param_name, param in lora_layer.named_parameters():
                        key = f"block_{index}.{attr_name}.{param_name}"
                        processor_state_dict[key] = param.data.clone()

    # torch.save({"processor_state_dict": processor_state_dict}, checkpoint_path)
    # print(f"Processor state_dict saved to {checkpoint_path}")
    return processor_state_dict

def load_processor_state_dict(model, processor_state_dict):
    """Load trainable parameters of processors from a checkpoint."""
    for index, block in enumerate(model.transformer_blocks):
        if hasattr(block.attn1, "processor"):
            processor = block.attn1.processor
            for attr_name in ["to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"]:
                if hasattr(processor, attr_name):
                    lora_layer = getattr(processor, attr_name)
                    for param_name, param in lora_layer.named_parameters():
                        key = f"block_{index}.{attr_name}.{param_name}"
                        if key in processor_state_dict:
                            param.data.copy_(processor_state_dict[key])
                        else:
                            raise KeyError(f"Missing key {key} in checkpoint.")

# Prepare training parameters
def get_processor_params(processor):
    params = []
    for attr_name in ["to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"]:
        if hasattr(processor, attr_name):
            lora_layer = getattr(processor, attr_name)
            params.extend(p for p in lora_layer.parameters() if p.requires_grad)
    return params

def get_all_processor_params(transformer):
    all_params = []
    for block in transformer.transformer_blocks:
        if hasattr(block.attn1, "processor"):
            processor = block.attn1.processor
            all_params.extend(get_processor_params(processor))
    return all_params