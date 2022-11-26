# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import pytest
import torch

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder

testcases = [
    {},
    {"vocab_size": 64000},
    {"activation_fn": "relu"},
    {"drop_path_rate": 0.1},
    {"decoder_normalize_before": False},
    {"no_scale_embedding": False},
    {"layernorm_embedding": True},
    {"rel_pos_buckets": 32, "max_rel_pos": 256},
    {"deepnorm": True, "subln": False, "decoder_normalize_before": False},
    {"bert_init": True},
    {"multiway": True},
    {"share_decoder_input_output_embed": True},
    {"checkpoint_activations": True},
    {"fsdp": True},
]


@pytest.mark.parametrize("args", testcases)
def test_decoder(args):
    config = DecoderConfig(**args)
    model = Decoder(config)
    prev_output_tokens = torch.ones(2, 10)
    token_embeddings = torch.rand(2, 10, config.decoder_embed_dim)
    model(
        prev_output_tokens=prev_output_tokens,
        token_embeddings=token_embeddings,
        features_only=True,
    )
