# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper, wrap

from torchscale.architecture.utils import init_bert_params
from torchscale.component.droppath import DropPath
from torchscale.component.feedforward_network import make_experts
from torchscale.component.gate_linear_unit import GLU
from torchscale.component.multiscale_retention import MultiScaleRetention
from torchscale.component.xmoe.moe_layer import MOELayer
from torchscale.component.xmoe.routing import Top1Gate, Top2Gate
from torchscale.component.rms_norm import RMSNorm
    
    
class RetNetRelPos(nn.Module):
    def __init__(self, args):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, args.decoder_embed_dim // args.decoder_retention_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.decoder_retention_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = args.recurrent_chunk_size
        
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            
            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class DecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        depth,
        is_moe_layer=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = self.build_retention(self.embed_dim, args)

        self.normalize_before = args.decoder_normalize_before

        self.retention_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = self.build_ffn(
                self.embed_dim,
                self.args,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)

        self.final_layer_norm = RMSNorm(self.embed_dim, eps=args.layernorm_eps)

        if args.deepnorm:
            self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return GLU(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
        )

    def build_retention(self, embed_dim, args):
        return MultiScaleRetention(
            args,
            embed_dim,
            args.decoder_value_embed_dim,
            args.decoder_retention_heads,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        x,
        incremental_state=None,
        chunkwise_recurrent=False,
        retention_rel_pos=None,
    ):
        residual = x
        if self.normalize_before:
            x = self.retention_layer_norm(x)

        x = self.retention(
            x,
            incremental_state=incremental_state,
            rel_pos=retention_rel_pos,
            chunkwise_recurrent=chunkwise_recurrent,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.retention_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x, l_aux = self.moe_layer(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, l_aux


class RetNetDecoder(nn.Module):
    def __init__(
        self,
        args,
        embed_tokens=None,
        output_projection=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.args = args

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens

        if (
            output_projection is None
            and not args.no_output_layer
            and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                )
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(args)
        self.chunkwise_recurrent = args.chunkwise_recurrent
        self.recurrent_chunk_size = args.recurrent_chunk_size
        

        if args.deepnorm:
            init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

    def build_output_projection(
        self,
        args,
    ):
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )
        return output_projection

    def build_decoder_layer(
        self, args, depth, is_moe_layer=False
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
    ):
        if incremental_state is not None and not self.is_first_step(incremental_state):
            tokens = tokens[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    def forward(
        self,
        prev_output_tokens,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        **kwargs
    ):
        # embed tokens
        x, _ = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )
        is_first_step = self.is_first_step(incremental_state)

        
        if self.chunkwise_recurrent and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0:
            padding_len = self.recurrent_chunk_size - prev_output_tokens.size(1) % self.recurrent_chunk_size
            slen = prev_output_tokens.size(1) + padding_len
            x = F.pad(x, (0, 0, 0, padding_len))
        else:
            slen = prev_output_tokens.size(1)
        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen, incremental_state is not None and not is_first_step, chunkwise_recurrent=self.chunkwise_recurrent)
        # decoder layers
        inner_states = [x]

        l_aux = []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                if idx not in incremental_state:
                    incremental_state[idx] = {}
                    
            x, l_aux_i = layer(
                x,
                incremental_state[idx] if incremental_state is not None else None,
                retention_rel_pos=retention_rel_pos,
                chunkwise_recurrent=self.chunkwise_recurrent,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)
            
        if self.chunkwise_recurrent and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0:
            x = x[:, :prev_output_tokens.size(1), :]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_layer(x)

        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }

    def output_layer(self, features):
        return self.output_projection(features)
