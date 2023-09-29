# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from fairseq import distributed_utils, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
from omegaconf import II

from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

DEFAULT_MAX_TARGET_POSITIONS = 1024
logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig(FairseqDataclass):
    activation_fn: str = field(
        default="swish", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_value_embed_dim: int = field(
        default=864, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=864, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_retention_heads: int = field(
        default=2, metadata={"help": "num decoder retention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply norm before each decoder block"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add norm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        },
    )
    moe_freq: int = field(
        default=0,
        metadata={"help": "Frequency at which we insert MoE Transformer layers"},
    )
    moe_expert_count: int = field(
        default=0, metadata={"help": "Number of experts in each MoE Layer"}
    )
    moe_gating_use_fp32: bool = field(
        default=False,
        metadata={"help": "Use FP32 computations in MoE top2 gating function"},
    )
    moe_second_expert_policy: str = field(
        default="sampling",
        metadata={"help": "policy for second expert, options: all/sampling/random"},
    )
    moe_normalize_gate_prob_before_dropping: bool = field(
        default=False,
        metadata={
            "help": "whether to normalize gate probs before or after dropping experts for capacity and randomization"
        },
    )
    moe_expert_ffn_dim: Optional[int] = field(
        default=None, metadata={"help": "MoE expert FFN dimension"}
    )
    moe_top1_expert: Optional[bool] = field(
        default=False, metadata={"help": "Use top1 gate instead of top2"}
    )
    moe_eval_capacity_token_fraction: Optional[float] = field(
        default=0.25,
        metadata={
            "help": (
                "Default: 0.25, Fraction of tokens as capacity during validation, "
                "if set to negative, use same as training. range: (0.0, 1.0]."
            )
        },
    )
    moe_normalize_expert_grad: Optional[str] = field(
        default="world_size",
        metadata={
            "help": "Divide expert gradients by (1) 'world_size' (2) 'sqrt_world_size'"
        },
    )
    record_a2a_perf_stats: Optional[bool] = field(
        default=False,
        metadata={"help": "records all to all perf stats during distributed training"},
    )
    dummy_a2a: Optional[bool] = field(
        default=False,
        metadata={
            "help": "By passes all to all during distributed training by returning the input buffer as output"
        },
    )
    moe_batch_prioritized_routing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if true orders token by the gate prob before capacity dropping."
        },
    )
    use_xmoe: Optional[bool] = field(
        default=False,
    )
    chunkwise_recurrent: Optional[bool] = field(
        default=False,
    )
    recurrent_chunk_size: Optional[int] = field(
        default=512,
    )
    

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    ddp_rank: int = II("distributed_training.distributed_rank")
    deepnorm: Optional[bool] = field(
        default=False,
    )
    subln: Optional[bool] = field(
        default=False,
    )


@register_model("retnet", dataclass=LanguageConfig)
class RetNetLanguageModel(FairseqLanguageModel):
    def __init__(self, args, decoder):
        self.args = args
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_embed_dim
        )
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, len(task.dictionary), bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim**-0.5
            )

        if getattr(args, "moe_freq", 0) > 0 and (
            getattr(args, "fp16", False)
            and not getattr(args, "memory_efficient_fp16", False)
            and getattr(args, "ddp_backend", None) != "fully_sharded"
        ):
            assert (
                args.fp16_no_flatten_grads
            ), "If training moe models, set --fp16-no-flatten-grads to calculate correct gradnorm"

        args.ddp_rank = distributed_utils.get_data_parallel_rank()

        config = RetNetConfig()
        config.override(args)

        decoder = LMDecoder(
            config,
            embed_tokens,
            output_projection,
            dictionary=task.dictionary,
        )

        return cls(args, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return Embedding(len(dictionary), embed_dim, dictionary.pad())


class LMDecoder(RetNetDecoder, FairseqIncrementalDecoder):
    def forward(self, src_tokens, **kwargs):
        return super().forward(src_tokens, **kwargs)

    def max_positions(self):
        return self.args.max_target_positions

    def reorder_incremental_state_scripting(
        self,
        incremental_state,
        new_order,
    ):
        for module in incremental_state:
            for key in incremental_state[module]:
                result = incremental_state[module][key].index_select(0, new_order)
                incremental_state[module][key] = result


@register_model_architecture("retnet", "retnet_base")
def retnet_base_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 864)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 864)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 2)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "swish")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)

    args.base_layers = getattr(args, "base_layers", 0)
    args.base_sublayers = getattr(args, "base_sublayers", 1)
    args.base_shuffle = getattr(args, "base_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    
    args.chunkwise_recurrent = getattr(args, "chunkwise_recurrent", False)
    args.recurrent_chunk_size = getattr(args, "recurrent_chunk_size", 512)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

@register_model_architecture("retnet", "retnet_medium")
def retnet_medium(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 1728)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1728)
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 4)
    retnet_base_architecture(args)
    
@register_model_architecture("retnet", "retnet_xl")
def retnet_xl(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 3456)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3456)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    retnet_base_architecture(args)

@register_model_architecture("retnet", "retnet_3b")
def retnet_3b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2560)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 4280)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4280)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 10)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    retnet_base_architecture(args)

@register_model_architecture("retnet", "retnet_7b")
def retnet_7b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 6912)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6912)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    retnet_base_architecture(args)

@register_model_architecture("retnet", "retnet_13b")
def retnet_13b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 5120)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 8560)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8560)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 20)
    args.decoder_layers = getattr(args, "decoder_layers", 40)
    retnet_base_architecture(args)

@register_model_architecture("retnet", "retnet_65b")
def retnet_65b(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 8192)
    args.decoder_value_embed_dim = getattr(args, "decoder_value_embed_dim", 13824)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 13824)
    args.decoder_retention_heads = getattr(args, "decoder_retention_heads", 32)
    args.decoder_layers = getattr(args, "decoder_layers", 64)
    retnet_base_architecture(args)
    
