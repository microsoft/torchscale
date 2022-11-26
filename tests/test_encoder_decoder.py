# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import pytest
import torch

from torchscale.architecture.config import EncoderDecoderConfig
from torchscale.architecture.encoder_decoder import EncoderDecoder
from torchscale.component.embedding import PositionalEmbedding, TextEmbedding

testcases = [
    {},
    {"vocab_size": 64000},
    {"activation_fn": "relu"},
    {"drop_path_rate": 0.1},
    {"encoder_normalize_before": False, "decoder_normalize_before": False},
    {"no_scale_embedding": False},
    {"layernorm_embedding": True},
    {"rel_pos_buckets": 32, "max_rel_pos": 256},
    {
        "deepnorm": True,
        "subln": False,
        "encoder_normalize_before": False,
        "decoder_normalize_before": False,
    },
    {"bert_init": True},
    {"multiway": True},
    {"share_decoder_input_output_embed": True},
    {"share_all_embeddings": True},
    {"checkpoint_activations": True},
    {"fsdp": True},
]


@pytest.mark.parametrize("args", testcases)
def test_decoder(args):
    config = EncoderDecoderConfig(**args)
    model = EncoderDecoder(
        config,
        encoder_embed_tokens=TextEmbedding(64000, config.encoder_embed_dim),
        decoder_embed_tokens=TextEmbedding(64000, config.decoder_embed_dim),
        encoder_embed_positions=PositionalEmbedding(
            config.max_source_positions, config.encoder_embed_dim
        ),
        decoder_embed_positions=PositionalEmbedding(
            config.max_target_positions, config.decoder_embed_dim
        ),
    )

    src_tokens = torch.ones(2, 20).long()
    prev_output_tokens = torch.ones(2, 10).long()

    model(
        src_tokens=src_tokens,
        prev_output_tokens=prev_output_tokens,
        features_only=True,
    )
