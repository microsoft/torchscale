# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import pytest
import torch

from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder

testcases = [
    {},
    {"vocab_size": 64000},
    {"activation_fn": "relu"},
    {"drop_path_rate": 0.1},
    {"encoder_normalize_before": False},
    {"no_scale_embedding": False},
    {"layernorm_embedding": True},
    {"rel_pos_buckets": 32, "max_rel_pos": 256},
    {"deepnorm": True, "subln": False, "encoder_normalize_before": False},
    {"bert_init": True},
    {"multiway": True},
    {"share_encoder_input_output_embed": True},
    {"checkpoint_activations": True},
    {"fsdp": True},
]


@pytest.mark.parametrize("args", testcases)
def test_encoder(args):
    config = EncoderConfig(**args)
    model = Encoder(config)
    token_embeddings = torch.rand(2, 10, config.encoder_embed_dim)
    model(src_tokens=None, token_embeddings=token_embeddings)
