import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from safetensors.torch import load_file

logger = logging.get_logger(__name__)

@torch.no_grad()
def decode_latents(pipe, latents):
    video = pipe.decode_latents(latents)
    video = pipe.video_processor.postprocess_video(video=video, output_type="np")
    return video

def create_attention_mask(text_length: int, seq_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create an attention mask to block text from attending to alpha.

    Args:
        text_length: Length of the text sequence.
        seq_length: Length of the other sequence.
        device: The device where the mask will be stored.
        dtype: The data type of the mask tensor.

    Returns:
        An attention mask tensor.
    """
    total_length = text_length + seq_length
    dense_mask = torch.ones((total_length, total_length), dtype=torch.bool)
    dense_mask[:text_length, text_length + seq_length // 2:] = False
    return dense_mask.to(device=device, dtype=dtype)

class RGBALoRACogVideoXAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. 
    It applies a rotary embedding on query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, device, dtype, attention_mask, lora_rank=128, lora_alpha=1.0, latent_dim=3072):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0 or later.")

        # Initialize LoRA layers
        self.lora_alpha = lora_alpha
        self.lora_rank = lora_rank

        # Helper function to create LoRA layers
        def create_lora_layer(in_dim, mid_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, mid_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(mid_dim, out_dim, bias=False, device=device, dtype=dtype)
            )

        self.to_q_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_k_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_v_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_out_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)

        # Store attention mask
        self.attention_mask = attention_mask

    def _apply_lora(self, hidden_states, seq_len, query, key, value, scaling):
        """Applies LoRA updates to query, key, and value tensors."""
        query_delta = self.to_q_lora(hidden_states).to(query.device)
        query[:, -seq_len // 2:, :] += query_delta[:, -seq_len // 2:, :] * scaling

        key_delta = self.to_k_lora(hidden_states).to(key.device)
        key[:, -seq_len // 2:, :] += key_delta[:, -seq_len // 2:, :] * scaling

        value_delta = self.to_v_lora(hidden_states).to(value.device)
        value[:, -seq_len // 2:, :] += value_delta[:, -seq_len // 2:, :] * scaling

        return query, key, value

    def _apply_rotary_embedding(self, query, key, image_rotary_emb, seq_len, text_seq_length, attn):
        """Applies rotary embeddings to query and key tensors."""
        from diffusers.models.embeddings import apply_rotary_emb

        # Apply rotary embedding to RGB and alpha sections
        query[:, :, text_seq_length:text_seq_length + seq_len // 2] = apply_rotary_emb(
            query[:, :, text_seq_length:text_seq_length + seq_len // 2], image_rotary_emb)
        query[:, :, text_seq_length + seq_len // 2:] = apply_rotary_emb(
            query[:, :, text_seq_length + seq_len // 2:], image_rotary_emb)
        
        if not attn.is_cross_attention:
            key[:, :, text_seq_length:text_seq_length + seq_len // 2] = apply_rotary_emb(
                key[:, :, text_seq_length:text_seq_length + seq_len // 2], image_rotary_emb)
            key[:, :, text_seq_length + seq_len // 2:] = apply_rotary_emb(
                key[:, :, text_seq_length + seq_len // 2:], image_rotary_emb)

        return query, key

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Concatenate encoder and decoder hidden states
        text_seq_length = encoder_hidden_states.size(1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape
        seq_len = hidden_states.shape[1] - text_seq_length
        scaling = self.lora_alpha / self.lora_rank

        # Apply LoRA to query, key, value
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query, key, value = self._apply_lora(hidden_states, seq_len, query, key, value, scaling)

        # Reshape query, key, value for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Normalize query and key if required
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            query, key = self._apply_rotary_embedding(query, key, image_rotary_emb, seq_len, text_seq_length, attn)

        # Compute scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=self.attention_mask, dropout_p=0.0, is_causal=False
        )

        # Reshape the output tensor back to the original shape
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # Apply linear projection and LoRA to the output
        original_hidden_states = attn.to_out[0](hidden_states)
        hidden_states_delta = self.to_out_lora(hidden_states).to(hidden_states.device)
        original_hidden_states[:, -seq_len // 2:, :] += hidden_states_delta[:, -seq_len // 2:, :] * scaling

        # Apply dropout
        hidden_states = attn.to_out[1](original_hidden_states)

        # Split back into encoder and decoder hidden states
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        return hidden_states, encoder_hidden_states

def prepare_for_rgba_inference(
    model, rgba_weights_path: str, device: torch.device, dtype: torch.dtype,
    lora_rank: int = 128, lora_alpha: float = 1.0, text_length: int = 226, seq_length: int = 35100
):
    def load_lora_sequential_weights(lora_layer, lora_layers, prefix):
        lora_layer[0].load_state_dict({'weight': lora_layers[f"{prefix}.lora_A.weight"]})
        lora_layer[1].load_state_dict({'weight': lora_layers[f"{prefix}.lora_B.weight"]})


    rgba_weights = load_file(rgba_weights_path)
    aux_emb = rgba_weights['domain_emb']

    attention_mask = create_attention_mask(text_length, seq_length, device, dtype)
    attn_procs = {}

    for name in model.attn_processors.keys():
        attn_processor = RGBALoRACogVideoXAttnProcessor(
            device=device, dtype=dtype, attention_mask=attention_mask,
            lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        
        index = name.split('.')[1]
        base_prefix = f'transformer.transformer_blocks.{index}.attn1'

        for lora_layer, prefix in [
            (attn_processor.to_q_lora, f'{base_prefix}.to_q'),
            (attn_processor.to_k_lora, f'{base_prefix}.to_k'),
            (attn_processor.to_v_lora, f'{base_prefix}.to_v'),
            (attn_processor.to_out_lora, f'{base_prefix}.to_out.0'),
        ]:
            load_lora_sequential_weights(lora_layer, rgba_weights, prefix)

        attn_procs[name] = attn_processor

    model.set_attn_processor(attn_procs)

    def custom_forward(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: Union[int, float, torch.LongTensor],
            timestep_cond: Optional[torch.Tensor] = None,
            ofs: Optional[Union[int, float, torch.LongTensor]] = None,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ):
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

            batch_size, num_frames, channels, height, width = hidden_states.shape

            # 1. Time embedding
            timesteps = timestep
            t_emb = self.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=hidden_states.dtype)
            emb = self.time_embedding(t_emb, timestep_cond)

            if self.ofs_embedding is not None:
                ofs_emb = self.ofs_proj(ofs)
                ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
                ofs_emb = self.ofs_embedding(ofs_emb)
                emb = emb + ofs_emb

            # 2. Patch embedding
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
            hidden_states = self.embedding_dropout(hidden_states)

            text_seq_length = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]

            hidden_states[:, hidden_states.size(1) // 2:, :] += aux_emb.expand(batch_size, -1, -1).to(hidden_states.device, dtype=hidden_states.dtype)

            # 3. Transformer blocks
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
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

            if not self.config.use_rotary_positional_embeddings:
                # CogVideoX-2B
                hidden_states = self.norm_final(hidden_states)
            else:
                # CogVideoX-5B
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states = self.norm_final(hidden_states)
                hidden_states = hidden_states[:, text_seq_length:]

            # 4. Final block
            hidden_states = self.norm_out(hidden_states, temb=emb)
            hidden_states = self.proj_out(hidden_states)

            # 5. Unpatchify
            p = self.config.patch_size
            p_t = self.config.patch_size_t

            if p_t is None:
                output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
                output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output = hidden_states.reshape(
                    batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)


        return forward

    model.forward = custom_forward(model)

