# Fine-tuning LongViT on TCGA Survival Prediction

## Setup

1. Download TCGA diagnostic whole slides from [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/), and organize the dataset (e.g., BRCA WSIs) as following structure:

```
/path/to/your_WSIs/
  TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs           
  ...
  TCGA-4H-AAAK-01Z-00-DX1.ABF1B042-1970-4E28-8671-43AAD393D2F9.svs
  ...       
```

2. Download [dataset annotation csv](https://github.com/mahmoodlab/MCAT/tree/master/datasets_csv_sig) and [splits for cross validation](https://github.com/mahmoodlab/MCAT/tree/master/splits/5foldcv) from the MCAT repository.

3. Generate the index json files of each split using the following command.
```
# Modify the `csv_path` and `csv_split_path` to your path.
python data_preprocessing/create_tcga_survival_index.py
```

4. Resize whole slide images to the desired size for finetuning.
```
python data_preprocessing/convert_wsi_to_images.py /path/to/your_WSIs /path/to/your_resized_WSIs ${target_size} ${wsi_level}
```

5. (Optional) For very large images (e.g., 32,768x32,768), we suggest parallelizing the training across multiple GPU devices due to the constraints of computation and memory. We split the sequence of millions of patches along the sequence dimension.
```
# num_splits is equal to the number of GPUs you used (e.g., 8 in our experiment) 
python data_preprocessing/split_to_small_images.py /path/to/your_resized_WSIs /path/to/your_splited_WSIs --num_splits ${num_splits}
```


## Example: Fine-tuning LongViT on TCGA Survival Prediction

The LongViT model can be fine-tuned using 8 V100-32GB. For images with a size less than or equal to 16,384x16,384, we can directly perform finetuning without using sequence parallel.

```bash
# IMAGE_SIZE - {1024, 4096, 8192, 16384}
# TASK - {"brca", "kidney", "lung"}  
# K_FOLD - {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}   
python -m torch.distributed.launch --nproc_per_node=8 run_longvit_finetuning.py \
        --input_size ${IMAGE_SIZE} \
        --model longvit_small_patch32_${IMAGE_SIZE} \
        --task tcga_${TASK}_survival \
        --batch_size 1 \
        --layer_decay 1.0 \
        --lr 5e-5 \
        --update_freq 1 \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --finetune /your_longvit_model_path/longvit_small_patch32_1024.pth
        --data_path ./survival_split_index/tcga_${TASK} \
        --image_dir /path/to/your_resized_WSIs  \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --k_fold ${K_FOLD} \
        --num_workers 1 \
        --enable_deepspeed \
        --model_key teacher \
        --randaug
```
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretraining).
- `--randaug`: perform image augmentation.


Parallelize the training of 32,768x32,768 images:

```bash
# TASK - {"brca", "kidney", "lung"}  
# K_FOLD - {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}   
python -m torch.distributed.launch --nproc_per_node=8 run_longvit_finetuning.py \
        --input_size 32768 \
        --model longvit_small_patch32_32768 \
        --task tcga_${TASK}_survival \
        --batch_size 2 \
        --layer_decay 1.0 \
        --lr 5e-5 \
        --update_freq 4 \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --finetune /your_longvit_model_path/longvit_small_patch32_1024.pth
        --data_path ./subtyping_split_index/tcga_${TASK} \
        --image_dir /path/to/your_splited_WSIs  \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --k_fold ${K_FOLD} \
        --num_workers 1 \
        --enable_deepspeed \
        --model_key teacher \
        --seq_parallel
```
- `--finetune`: weight path of your pretrained models; please download the pretrained model weights in [README.md](../README.md#pretraining).
- `--seq_parallel`: parallelize the training for very large images.


## Example: Evaluate LongViT on TCGA Subtyping

```bash
# IMAGE_SIZE - {1024, 4096, 8192, 16384, 32768}
# TASK - {"brca", "kidney", "lung"}  
# K_FOLD - {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}  
python -m torch.distributed.launch --nproc_per_node=1 run_longvit_finetuning.py \
        --input_size ${IMAGE_SIZE} \
        --model longvit_small_patch32_${IMAGE_SIZE} \
        --task tcga_${TASK}_survival \
        --batch_size 1 \
        --layer_decay 1.0 \
        --lr 5e-5 \
        --update_freq 1 \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.1 \
        --finetune /path/to/save/your_model/checkpoint-best/mp_rank_00_model_states.pt \
        --data_path ./survival_split_index/tcga_${TASK} \
        --image_dir /path/to/your_resized_WSIs \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --k_fold ${K_FOLD} \
        --num_workers 1 \
        --enable_deepspeed \
        --model_key module \
        --eval \
        --no_auto_resume
```
- `--eval`: performing evaluation.
- `--finetune`: best val model.

For the model trained with sequence parallel, add `--seq_parallel` and use the same number of GPUs as training to perform evaluation.