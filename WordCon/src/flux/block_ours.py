import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora
import math

# Monkey patch Attention class
def set_controller(self, controller):
    self.controller = controller

Attention.controller = None
Attention.set_controller = set_controller

# refer to /path/attention-map-diffusers/attention_map_diffusers/modules.py
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight

def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
    
    # 只保存 mm-dit 中的部分 attention map
    if attn.training and hasattr(attn, 'controller') and attn.controller is not None and encoder_hidden_states is not None:
        # 计算 attention map
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        attention_probs = F.softmax(attention_scores, dim=-1)

        text_length = encoder_hidden_states_query_proj.shape[2]
        # (1,24,4608,4608) -> (1,24,4096,4096)
        attention_probs = attention_probs[:,:,text_length:,:text_length]  # 不调用 .cpu()
        
        is_double = encoder_hidden_states is not None
        place_in_transformer = "mm_dit" if is_double else "single_dit"
        attn.controller(attention_probs, is_double, place_in_transformer)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        if condition_latents is not None:
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[
                    :, encoder_hidden_states.shape[1] : -condition_latents.shape[1]
                ],
                hidden_states[:, -condition_latents.shape[1] :],
            )
        else:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        # with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None:
            condition_latents = attn.to_out[0](condition_latents)
            condition_latents = attn.to_out[1](condition_latents)

        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    elif condition_latents is not None:
        # if there are condition_latents, we need to separate the hidden_states and the condition_latents
        hidden_states, condition_latents = (
            hidden_states[:, : -condition_latents.shape[1]],
            hidden_states[:, -condition_latents.shape[1] :],
        )
        return hidden_states, condition_latents
    else:
        return hidden_states


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):
    use_cond = condition_latents is not None
    with enable_lora((self.norm1.linear,), model_config.get("latent_lora", False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    if use_cond:
        (
            norm_condition_latents,
            cond_gate_msa,
            cond_shift_mlp,
            cond_scale_mlp,
            cond_gate_mlp,
        ) = self.norm1(condition_latents, emb=cond_temb)

    # Attention.
    result = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
        condition_latents = condition_latents + cond_attn_output
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )
    # 3. condition_latents
    if use_cond:
        norm_condition_latents = self.norm2(condition_latents)
        norm_condition_latents = (
            norm_condition_latents * (1 + cond_scale_mlp[:, None])
            + cond_shift_mlp[:, None]
        )

    # Feed-forward.
    with enable_lora((self.ff.net[2],), model_config.get("latent_lora", False)):
        # 1. hidden_states
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    # 2. encoder_hidden_states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
    # 3. condition_latents
    if use_cond:
        cond_ff_output = self.ff(norm_condition_latents)
        cond_ff_output = cond_gate_mlp.unsqueeze(1) * cond_ff_output

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = condition_latents + cond_ff_output

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None


def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    condition_latents: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    cond_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):

    using_cond = condition_latents is not None
    residual = hidden_states
    with enable_lora(
        (
            self.norm.linear,
            self.proj_mlp,
        ),
        model_config.get("latent_lora", False),
    ):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = self.norm(condition_latents, emb=cond_temb)
        mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_condition_latents))

    attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **(
            {
                "condition_latents": norm_condition_latents,
                "cond_rotary_emb": cond_rotary_emb if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    if using_cond:
        attn_output, cond_attn_output = attn_output

    with enable_lora((self.proj_out,), model_config.get("latent_lora", False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    if using_cond:
        condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
        cond_gate = cond_gate.unsqueeze(1)
        condition_latents = cond_gate * self.proj_out(condition_latents)
        condition_latents = residual_cond + condition_latents

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states if not using_cond else (hidden_states, condition_latents)
