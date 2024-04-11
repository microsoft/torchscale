# [(LongViT) When an Image is Worth 1,024 x 1,024 Words: A Case Study in Computational Pathology](https://arxiv.org/abs/2312.03558)

**LongViT** is a vision Transformer that can process gigapixel images (e.g., 32,768x32,768 images) in an end-to-end manner. We split the image into millions of patches and employ [LongNet](https://arxiv.org/abs/2307.02486) to directly model the extremely long sequence. We apply LongViT in the field of computational pathology and achieve remarkable performance on cancer subtyping and survival prediction tasks.


## Setup
```
pip install -r requirements.txt
pip install git+https://github.com/shumingma/fairseq.git@moe
pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.20#egg=xformers
```


## Pretraining

We perform self-supervised pretraining on TCGA diagnostic slides using [DINO](https://arxiv.org/abs/2104.14294) objective. The detailed instructions can be found at [`get_started_for_tcga_pretraining.md`](get_started/get_started_for_tcga_pretraining.md).

The link to the pretrained LongViT model on TCGA diagnostic slides:
   - [`LongViT`](https://github.com/wenhui0924/longvit_ckpts/releases/download/longvit/longvit_small_patch32_1024.pth): #layer=12; hidden=384; FFN factor=4x; #head=16; patch=32x32


## Fine-tuning on Subtyping Classification

We perform finetuning on cancer subtyping on images with sizes up to 32,768x32,768 (1M patches). The detailed instructions can be found at [`get_started_for_tcga_subtyping.md`](get_started/get_started_for_tcga_subtyping.md).


## Fine-tuning on Survival Prediction

We perform finetuning on survival prediction on images with sizes up to 32,768x32,768 (1M patches). The detailed instructions can be found at [`get_started_for_tcga_survival_prediction.md`](get_started/get_started_for_tcga_survival_prediction.md).


## Citation

If you find this repository useful, please consider citing our work:
```
@article{longvit,
  title={When an Image is Worth 1,024 x 1,024 Words: A Case Study in Computational Pathology},
  author={Wang, Wenhui and Ma, Shuming and Xu, Hanwen and Usuyama, Naoto and Ding, Jiayu and Poon, Hoifung and Wei, Furu},
  journal={arXiv preprint arXiv:2312.03558},
  year={2023}
}

@article{longnet,
  title={LongNet: Scaling transformers to 1,000,000,000 tokens},
  author={Ding, Jiayu and Ma, Shuming and Dong, Li and Zhang, Xingxing and Huang, Shaohan and Wang, Wenhui and Zheng, Nanning and Wei, Furu},
  journal={arXiv preprint arXiv:2307.02486},
  year={2023}
}

@article{torchscale,
  title={TorchScale: Transformers at scale},
  author={Ma, Shuming and Wang, Hongyu and Huang, Shaohan and Wang, Wenhui and Chi, Zewen and Dong, Li and Benhaim, Alon and Patra, Barun and Chaudhary, Vishrav and Song, Xia and others},
  journal={arXiv preprint arXiv:2211.13184},
  year={2022}
}
```


## Acknowledgement

This repository is built using the [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3), the [MCAT](https://github.com/mahmoodlab/MCAT), the [DINO](https://github.com/facebookresearch/dino), the [HIPT](https://github.com/mahmoodlab/HIPT) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using LongViT models, please submit a GitHub issue.
