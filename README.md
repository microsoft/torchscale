# TorchScale - A Library of Foundation Architectures

<p>
  <a href="https://github.com/microsoft/torchscale/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/torchscale"><img alt="MIT License" src="https://badge.fury.io/py/torchscale.svg" /></a>
</p>

TorchScale is a PyTorch library that allows researchers and developers to scale up Transformers efficiently and effectively.

Fundamental research to develop new architectures for foundation models and A(G)I, focusing on modeling generality and capability, as well as training stability and efficiency.
- Stability - [**DeepNet**](https://arxiv.org/abs/2203.00555): scaling Transformers to 1,000 Layers and beyond
- Generality - [**Foundation Transformers (Magneto)**](https://arxiv.org/abs/2210.06423): towards true general-purpose modeling across tasks and modalities (including language, vision, speech, and multimodal)
- Capability - A [**Length-Extrapolatable**](https://arxiv.org/abs/2212.10554) Transformer
- Efficiency - [**X-MoE**](https://arxiv.org/abs/2204.09179): scalable & finetunable sparse Mixture-of-Experts (MoE)

### The Revolution of Model Architecture
- [**BitNet**](https://arxiv.org/abs/2310.11453): 1-bit Transformers for Large Language Models
- [**RetNet**](https://arxiv.org/abs/2307.08621): Retentive Network: A Successor to Transformer for Large Language Models
- [**LongNet**](https://arxiv.org/abs/2307.02486): Scaling Transformers to 1,000,000,000 Tokens

## News

- December, 2023: [LongNet](torchscale/model/LongNet.py) and [LongViT](examples/longvit/README.md) released
- October, 2023: Update RMSNorm and SwiGLU as the default module in RetNet
- November, 2022: TorchScale 0.1.1 released [[Paper](https://arxiv.org/abs/2211.13184)] [[PyPI](https://pypi.org/project/torchscale/)]

## Installation

To install:
```
pip install torchscale
```

Alternatively, you can develop it locally:
```
git clone https://github.com/microsoft/torchscale.git
cd torchscale
pip install -e .
```

For faster training install [Flash Attention](https://github.com/Dao-AILab/flash-attention) for Turing, Ampere, Ada, or Hopper GPUs:
```
pip install flash-attn
```
or [xFormers](https://github.com/facebookresearch/xformers) for Volta, Turing, Ampere, Ada, or Hopper GPUs:
```
# cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
# cuda 12.1 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

## Getting Started

It takes only several lines of code to create a model with the above fundamental research features enabled. Here is how to quickly obtain a BERT-like encoder:

```python
>>> from torchscale.architecture.config import EncoderConfig
>>> from torchscale.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000)
>>> model = Encoder(config)

>>> print(model)
```

We also support the `Decoder` architecture and the `EncoderDecoder` architecture:

```python
# Creating a decoder model
>>> from torchscale.architecture.config import DecoderConfig
>>> from torchscale.architecture.decoder import Decoder

>>> config = DecoderConfig(vocab_size=64000)
>>> decoder = Decoder(config)
>>> print(decoder)

# Creating a encoder-decoder model
>>> from torchscale.architecture.config import EncoderDecoderConfig
>>> from torchscale.architecture.encoder_decoder import EncoderDecoder

>>> config = EncoderDecoderConfig(vocab_size=64000)
>>> encdec = EncoderDecoder(config)
>>> print(encdec)
```

It takes only several lines of code to create a RetNet model:

```python
# Creating a RetNet model
>>> import torch
>>> from torchscale.architecture.config import RetNetConfig
>>> from torchscale.architecture.retnet import RetNetDecoder

>>> config = RetNetConfig(vocab_size=64000)
>>> retnet = RetNetDecoder(config)

>>> print(retnet)
```

For LongNet models ([Flash Attention](https://github.com/Dao-AILab/flash-attention) required):
```python
>>> import torch
>>> from torchscale.architecture.config import EncoderConfig, DecoderConfig
>>> from torchscale.model.longnet import LongNetEncoder, LongNetDecoder

# Creating a LongNet encoder with the dilated pattern of segment_length=[2048,4096] and dilated_ratio=[1,2]
>>> config = EncoderConfig(vocab_size=64000, segment_length='[2048,4096]', dilated_ratio='[1,2]', flash_attention=True)
>>> longnet = LongNetEncoder(config)

# Creating a LongNet decoder with the dilated pattern of segment_length=[2048,4096] and dilated_ratio=[1,2]
>>> config = DecoderConfig(vocab_size=64000, segment_length='[2048,4096]', dilated_ratio='[1,2]', flash_attention=True)
>>> longnet = LongNetDecoder(config)
```

## Key Features

- [DeepNorm to improve the training stability of Post-LayerNorm Transformers](https://arxiv.org/abs/2203.00555)
  * enabled by setting *deepnorm=True* in the `Config` class. 
  * It adjusts both the residual connection and the initialization method according to the model architecture (i.e., encoder, decoder, or encoder-decoder).

- [SubLN for the model generality and the training stability](https://arxiv.org/abs/2210.06423)
  * enabled by *subln=True*. This is enabled by default. 
  * It introduces another LayerNorm to each sublayer and adjusts the initialization according to the model architecture.
  * Note that SubLN and DeepNorm cannot be used in one single model.

- [X-MoE: efficient and finetunable sparse MoE modeling](https://arxiv.org/abs/2204.09179)
  * enabled by *use_xmoe=True*. 
  * It replaces every *'moe_freq'* `FeedForwardNetwork` layers with the X-MoE layers.

- [Multiway architecture for multimodality](https://arxiv.org/abs/2208.10442)
  * enabled by *multiway=True*.
  * It provides a pool of Transformer's parameters used for different modalities.

- [Extrapolatable position embedding (Xpos)](https://arxiv.org/abs/2212.10554)
  * enabled by *xpos_rel_pos=True*.

- [Relative position bias](https://arxiv.org/abs/1910.10683)
  * enabled by adjusting *rel_pos_buckets* and *max_rel_pos*.

- [SparseClip: improving the gradient clipping for sparse MoE models](https://arxiv.org/abs/2211.13184)
  * we provide a [sample code](examples/fairseq/utils/sparse_clip.py) that can be easily adapted to the FairSeq (or other) repo.

- [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)
  * created by `config = RetNetConfig(vocab_size=64000)` and `retnet = RetNetDecoder(config)`.

- [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486)
  
Most of the features above can be used by simply passing the corresponding parameters to the config. For example:

```python
>>> from torchscale.architecture.config import EncoderConfig
>>> from torchscale.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000, deepnorm=True, multiway=True)
>>> model = Encoder(config)

>>> print(model)
```

## Examples

We have examples of how to use TorchScale in the following scenarios/tasks:

- Language

  * [Decoder/GPT](examples/fairseq/README.md#example-gpt-pretraining)

  * [Encoder-Decoder/Neural Machine Translation](examples/fairseq/README.md#example-machine-translation)

  * [Encoder/BERT](examples/fairseq/README.md#example-bert-pretraining)

- Vision

  * [LongViT](examples/longvit/README.md)

  * ViT/BEiT [In progress]

- Speech

- Multimodal

  * [Multiway Transformers/BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)

We plan to provide more examples regarding different tasks (e.g. vision pretraining and speech recognition) and various deep learning toolkits (e.g. [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)). Any comments or PRs are welcome!


## Acknowledgments

Some implementations in TorchScale are either adapted from or inspired by the [FairSeq](https://github.com/facebookresearch/fairseq) repository and the [UniLM](https://github.com/microsoft/unilm) repository.

## Citations

If you find this repository useful, please consider citing our work:

```
@article{torchscale,
  author    = {Shuming Ma and Hongyu Wang and Shaohan Huang and Wenhui Wang and Zewen Chi and Li Dong and Alon Benhaim and Barun Patra and Vishrav Chaudhary and Xia Song and Furu Wei},
  title     = {{TorchScale}: {Transformers} at Scale},
  journal   = {CoRR},
  volume    = {abs/2211.13184},
  year      = {2022}
}
```

```
@article{deepnet,
  author    = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Dongdong Zhang and Furu Wei},
  title     = {{DeepNet}: Scaling {Transformers} to 1,000 Layers},
  journal   = {CoRR},
  volume    = {abs/2203.00555},
  year      = {2022},
}
```

```
@article{magneto,
  author    = {Hongyu Wang and Shuming Ma and Shaohan Huang and Li Dong and Wenhui Wang and Zhiliang Peng and Yu Wu and Payal Bajaj and Saksham Singhal and Alon Benhaim and Barun Patra and Zhun Liu and Vishrav Chaudhary and Xia Song and Furu Wei},
  title     = {Foundation {Transformers}},
  journal   = {CoRR},
  volume    = {abs/2210.06423},
  year      = {2022}
}
```

```
@inproceedings{xmoe,
  title={On the Representation Collapse of Sparse Mixture of Experts},
  author={Zewen Chi and Li Dong and Shaohan Huang and Damai Dai and Shuming Ma and Barun Patra and Saksham Singhal and Payal Bajaj and Xia Song and Xian-Ling Mao and Heyan Huang and Furu Wei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=mWaYC6CZf5}
}
```

```
@article{retnet,
  author={Yutao Sun and Li Dong and Shaohan Huang and Shuming Ma and Yuqing Xia and Jilong Xue and Jianyong Wang and Furu Wei},
  title     = {Retentive Network: A Successor to {Transformer} for Large Language Models},
  journal   = {ArXiv},
  volume    = {abs/2307.08621},
  year      = {2023}
}
```

```
@article{longnet,
  author={Jiayu Ding and Shuming Ma and Li Dong and Xingxing Zhang and Shaohan Huang and Wenhui Wang and Nanning Zheng and Furu Wei},
  title     = {{LongNet}: Scaling Transformers to 1,000,000,000 Tokens},
  journal   = {ArXiv},
  volume    = {abs/2307.02486},
  year      = {2023}
}
```

```
@article{longvit,
  title     = {When an Image is Worth 1,024 x 1,024 Words: A Case Study in Computational Pathology},
  author    = {Wenhui Wang and Shuming Ma and Hanwen Xu and Naoto Usuyama and Jiayu Ding and Hoifung Poon and Furu Wei},
  journal   = {ArXiv},
  volume    = {abs/2312.03558},
  year      = {2023}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [Furu Wei](mailto:fuwei@microsoft.com) and [Shuming Ma](mailto:shumma@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos is subject to those third-party's policies.
