from datasets import TCGASubtypingDataset

tcga_task = "tcga_brca"
for k_fold in range(10):
    TCGASubtypingDataset.make_dataset_index(
        task=tcga_task,
        csv_path="./subtyping_dataset_csv/{}_subset.csv.zip".format(tcga_task),
        csv_split_path="./subtyping_splits/10foldcv_subtype/{}/splits_{}.csv".format(tcga_task, k_fold),
        k_fold=k_fold,
        index_path="./subtyping_split_index/{}".format(tcga_task),
        ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'],
        label_dict = {'IDC':0, 'ILC':1},
    )

tcga_task = "tcga_lung"
for k_fold in range(10):
    TCGASubtypingDataset.make_dataset_index(
        task=tcga_task,
        csv_path="./subtyping_dataset_csv/{}_subset.csv.zip".format(tcga_task),
        csv_split_path="./subtyping_splits/10foldcv_subtype/{}/splits_{}.csv".format(tcga_task, k_fold),
        k_fold=k_fold,
        index_path="./subtyping_split_index/{}".format(tcga_task),
        ignore=[],
        label_dict = {'LUAD':0, 'LUSC':1},
    )

tcga_task = "tcga_kidney"
for k_fold in range(10):
    TCGASubtypingDataset.make_dataset_index(
        task=tcga_task,
        csv_path="./subtyping_dataset_csv/{}_subset.csv.zip".format(tcga_task),
        csv_split_path="./subtyping_splits/10foldcv_subtype/{}/splits_{}.csv".format(tcga_task, k_fold),
        k_fold=k_fold,
        index_path="./subtyping_split_index/{}".format(tcga_task),
        ignore=[],
        label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2},
    )
