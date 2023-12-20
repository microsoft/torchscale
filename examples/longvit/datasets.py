# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import random
import torch
import glob
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

import utils
import openslide
import pandas as pd
import numpy as np

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path, split, transform, 
        task=None, k_fold=0,
    ):
        index_files = self.get_index_files(split, k_fold=k_fold, task=task)
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))


def df_prep(data, label_dict, ignore, label_col):
    if label_col != 'label':
        data['label'] = data[label_col].copy()

    mask = data['label'].isin(ignore)
    data = data[~mask]
    data.reset_index(drop=True, inplace=True)
    for i in data.index:
        key = data.loc[i, 'label']
        data.at[i, 'label'] = label_dict[key]

    return data


def get_split_from_df(slide_data, all_splits, prop=1.0, seed=1, split_key='train'):
    split = all_splits[split_key].str.rstrip('.svs')
    split = split.dropna().reset_index(drop=True)

    if len(split) > 0:
        mask = slide_data['slide_id'].isin(split.tolist())
        df_slice = slide_data[mask].reset_index(drop=True)
        if split_key == 'train' and prop != 1.0:
            df_slice = df_slice.sample(frac=prop, random_state=seed).reset_index(drop=True)
        if split_key == 'train':
            print(df_slice.head())
        print("Traing Data Size ({%0.2f}): %d" % (prop, df_slice.shape[0]))
    else:
        df_slice = None
    
    return df_slice


class TCGASubtypingDataset(BaseDataset):
    def __init__(self, data_path, split, transform, task, k_fold, image_dir, seq_parallel=False, cached_randaug=False):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, task=task, k_fold=k_fold, 
        )
        self.k_fold = k_fold
        self.image_dir = image_dir
        self.seq_parallel = seq_parallel
        self.cached_randaug = cached_randaug

    @staticmethod
    def get_index_files(split, k_fold=0, task=None):
        if split == "train":
            return ("{}.train.index.{}.jsonl".format(task.replace("_subtyping", ""), k_fold), )
        elif split == "val":
            return ("{}.val.index.{}.jsonl".format(task.replace("_subtyping", ""), k_fold), )
        elif split == "test":
            return ("{}.test.index.{}.jsonl".format(task.replace("_subtyping", ""), k_fold), )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        if self.seq_parallel:
            img_path = item["image_path"] + "_{}.jpg".format(utils.get_rank())
        else:
            img_path = item["image_path"] + ".jpg"
        img = self._get_image(img_path)
        data["image"] = img
        data["label"] = item["label"]
        return data

    def _get_image(self, image_path: str):
        if self.cached_randaug:
            if self.split == "train":
                cur_epoch = int(os.environ.get('cur_epoch'))
                image_path = os.path.join(self.image_dir, "epoch_{}".format(cur_epoch), image_path)
            else:
                image_path = os.path.join(self.image_dir, "wo_augmentation", image_path)
        else:
            image_path = os.path.join(self.image_dir, image_path)

        image = self.loader(image_path)
        return self.transform(image)

    @staticmethod
    def _make_tcga_index(task, csv_path, csv_split_path, k_fold, index_path, ignore, label_dict, split):
        items = []
        index_file = os.path.join(index_path, f"{task}.{split}.index.{k_fold}.jsonl")
        
        slide_data = pd.read_csv(csv_path)
        slide_data = df_prep(slide_data, label_dict, ignore, label_col="oncotree_code")
        slide_data['slide_id'] = slide_data['slide_id'].apply(lambda x: x.replace(".svs", ""))

        all_splits = pd.read_csv(csv_split_path)
        slide_data_split = get_split_from_df(slide_data, all_splits, split_key=split)
        
        for index, row in slide_data_split.iterrows():
            items.append({
                "image_path": row["slide_id"],
                "label": row["label"],
            })
            file_path = os.path.join(index_path.replace("tcga_", "") + "_svs", "{}.svs".format(row["slide_id"]))
            if not os.path.exists(file_path):
                print("file {} do not exists".format(row["slide_id"]))

        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, task, csv_path, csv_split_path, k_fold, index_path, ignore, label_dict):
        cls._make_tcga_index(
             task=task, csv_path=csv_path, csv_split_path=csv_split_path, k_fold=k_fold, index_path=index_path, 
             ignore=ignore, label_dict=label_dict, split="train",
        )
        cls._make_tcga_index(
             task=task, csv_path=csv_path, csv_split_path=csv_split_path, k_fold=k_fold, index_path=index_path, 
             ignore=ignore, label_dict=label_dict, split="val",
        )
        cls._make_tcga_index(
             task=task, csv_path=csv_path, csv_split_path=csv_split_path, k_fold=k_fold, index_path=index_path, 
             ignore=ignore, label_dict=label_dict, split="test",
        )


def get_survival_split_from_df(slide_data, all_splits, split_key='train'):
    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)

    if len(split) > 0:
        mask = slide_data['slide_id'].isin(split.tolist())
        df_slice = slide_data[mask].reset_index(drop=True)
    else:
        df_slice = None
    
    return df_slice


class TCGASurvivalDataset(BaseDataset):
    def __init__(self, data_path, split, transform, task, k_fold, image_dir, seq_parallel=False, cached_randaug=False):
        super().__init__(
            data_path=data_path, split=split, 
            transform=transform, task=task, k_fold=k_fold, 
        )
        self.k_fold = k_fold
        self.image_dir = image_dir
        self.seq_parallel = seq_parallel
        self.cached_randaug = cached_randaug

    @staticmethod
    def get_index_files(split, k_fold=0, task=None):
        if split == "train":
            return ("{}.train.index.{}.jsonl".format(task.replace("_survival", ""), k_fold), )
        elif split == "val":
            return ("{}.val.index.{}.jsonl".format(task.replace("_survival", ""), k_fold), )
        elif split == "test":
            return ("{}.val.index.{}.jsonl".format(task.replace("_survival", ""), k_fold), )
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        if self.seq_parallel:
            img_path = item["image_paths"][0].replace(".svs", "") + "_{}.jpg".format(utils.get_rank())
        else:
            img_path = item["image_paths"][0].replace(".svs", "") + ".jpg"
        img = self._get_image(img_path)
        case_id = item["case_id"]
        data["image"] = img
        data["label"] = item["label"]
        data["event_time"] = item["event_time"]
        data["censorship"] = item["censorship"]
        return data

    def _get_image(self, image_path: str):
        if self.cached_randaug:
            if self.split == "train":
                cur_epoch = int(os.environ.get('cur_epoch'))
                image_path = os.path.join(self.image_dir, "epoch_{}".format(cur_epoch), image_path)
            else:
                image_path = os.path.join(self.image_dir, "wo_augmentation", image_path)
        else:
            image_path = os.path.join(self.image_dir, image_path)

        image = self.loader(image_path)
        return self.transform(image)

    @staticmethod
    def _make_tcga_index(task, csv_path, csv_split_path, k_fold, index_path, split):
        items = []
        os.makedirs(index_path, exist_ok=True)
        index_file = os.path.join(index_path, f"{task}.{split}.index.{k_fold}.jsonl")

        slide_data = pd.read_csv(csv_path, low_memory=False)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        
        label_col = "survival_months"
        assert label_col in slide_data.columns
        
        # if "IDC" in slide_data['oncotree_code'].values: # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        n_bins = 4
        eps = 1e-6
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1).tolist()
            else:
                slide_ids = slide_ids.values.tolist()
            patient_dict.update({patient:slide_ids})

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        patients_df = slide_data.drop_duplicates(['case_id'])
        patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]

        all_splits = pd.read_csv(csv_split_path)
        slide_data_split = get_survival_split_from_df(slide_data, all_splits, split_key=split)
        
        for index, row in slide_data_split.iterrows():
            case_id = row["case_id"]
            items.append({
                "case_id" : row["case_id"],
                "label": row["disc_label"],
                "event_time": row["survival_months"],
                "censorship": row["censorship"], 
                "image_paths": patient_dict[case_id],
            })
            for slide_id in patient_dict[case_id]:
                file_path = os.path.join(f"/tmp/tcga/{task}_svs".replace("tcga_", ""), slide_id)
                if not os.path.exists(file_path):
                    print("file {} do not exists".format(row["slide_id"]))

        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, task, csv_path, csv_split_path, k_fold, index_path):
        cls._make_tcga_index(
             task=task, csv_path=csv_path, csv_split_path=csv_split_path, k_fold=k_fold, index_path=index_path, 
             split="train",
        )
        cls._make_tcga_index(
             task=task, csv_path=csv_path, csv_split_path=csv_split_path, k_fold=k_fold, index_path=index_path, 
             split="val",
        )


task2dataset = {
    "tcga_lung_subtyping": TCGASubtypingDataset,
    "tcga_kidney_subtyping": TCGASubtypingDataset,
    "tcga_brca_subtyping": TCGASubtypingDataset,
    "tcga_ucec_survival": TCGASurvivalDataset,
    "tcga_luad_survival": TCGASurvivalDataset,
    "tcga_brca_survival": TCGASurvivalDataset,
}


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, seq_parallel=False, seed=None):
    if is_train:
        if seq_parallel:
            generator = torch.Generator()  
            generator.manual_seed(seed)
            sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        else:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
            )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):

    if is_train:
        t = []
        if args.randaug:
            t += [
                 RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0), interpolation=args.train_interpolation), 
                 transforms.RandomHorizontalFlip(),
            ]

        t += [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            # transforms.Resize((args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return t


def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    print(transform)
    dataset_class = task2dataset[args.task]

    opt_kwargs = {}    
    if args.task.startswith("tcga"):
        opt_kwargs["k_fold"] = args.k_fold
        opt_kwargs["image_dir"] = args.image_dir
        opt_kwargs["seq_parallel"] = args.seq_parallel
        opt_kwargs["cached_randaug"] = args.cached_randaug

    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        transform=transform, task=args.task, **opt_kwargs, 
    )
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.0)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem,
        seq_parallel=args.seq_parallel, seed=args.seed,
    )


def create_downstream_dataset(args, is_eval=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)
    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
            create_dataset_by_split(args, split="val", is_train=False),
