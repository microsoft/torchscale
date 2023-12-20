from datasets import TCGASurvivalDataset

for tcga_task in ["tcga_ucec", "tcga_luad", "tcga_brca"]:
    for k_fold in range(5):
        TCGASurvivalDataset.make_dataset_index(
            task=tcga_task,
            csv_path="./survival_dataset_csv/{}_all_clean.csv.zip".format(tcga_task),
            csv_split_path="./survival_splits/5foldcv/{}/splits_{}.csv".format(tcga_task, k_fold),
            k_fold=k_fold,
            index_path="./survival_split_index/{}".format(tcga_task),
        )