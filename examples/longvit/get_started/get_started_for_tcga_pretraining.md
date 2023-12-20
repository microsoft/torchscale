# Pretraining LongViT on TCGA using DINO

## Setup

1. Download TCGA diagnostic whole slides from [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/).

2. Generate 1,024x1,024 regions from WSIs:
```
# we randomly generate 100 small regions for each whole slide image
python data_preprocessing/generate_1024_crops.py /path/to/your_WSIs /path/to/your_crops 100
```

## Pretraining LongViT

Replace the `vision_transformer.py` in [DINO](https://github.com/facebookresearch/dino) with [LongViT vision_transformer.py](../pretraining/vision_transformer.py), and modify the `global crop size` to 1024 and `local crop size` to 512 to preform LongViT pretraining using DINO framework.