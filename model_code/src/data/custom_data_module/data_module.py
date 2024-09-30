import os
import albumentations as A
from sklearn.model_selection import KFold
from src.data.Base_datamodule import BaseDataModule
from src.utils.data_utils import load_yaml_config
from typing import Optional
import torch
from torchvision import transforms
from src.data.datasets.dataset import CustomDataset
from src.data.collate_fns.collate_fn import collate_fn
from torch.utils.data import Subset

'''
    해당 함수는 데이터를 받아와 train / test 별 전처리가 가능합니다.
    또한 config의 증강을 활용하여 train 데이터에 대한 증강 처리를 지원합니다.

    Args : data_config_path, augmentation_config_path, seed

    Retrun : train_loader, test_loader, augmentation_transform 등 각 함수에 따라 다르게 적용

'''
class DataModule(BaseDataModule):
    def __init__(self, data_config_path: str, augmentation_config_path: str, seed: int):
        self.data_config = load_yaml_config(data_config_path)
        self.augmentation_config = load_yaml_config(augmentation_config_path)
        self.seed = seed
        self.n_splits = self.data_config['data']['k_folds']
        self.current_fold = 0
        self.kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.train_dataset = None
        self.val_dataset = None
        super().__init__(self.data_config)



    def setup(self, stage: Optional[str] = None):
        torch.manual_seed(self.seed)

        if self.augmentation_config["augmentation"]['use_augmentation']:
            train_transforms = self._get_augmentation_transforms()
        else:
            train_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Train 이미지 정규화 추가
            ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Test 이미지 정규화 추가
        ])

        raw_data_path = self.data_config['data']['raw_data_path']


        # Load the full dataset for KFold splitting
        self.dataset = CustomDataset(data_dir=raw_data_path, train=True, transform=train_transforms)

        # Test dataset stays the same
        self.test_dataset = CustomDataset(
            data_dir=raw_data_path, train=False, transform=test_transforms
        )


        # Initialize KFold with the specified number of splits
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        # Get the indices for the current fold
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            if fold_idx == self.current_fold:
                # Create training and validation subsets for the current fold
                self.train_dataset = Subset(self.dataset, train_idx)
                self.val_dataset = Subset(self.dataset, val_idx)
                break

    def set_fold(self, fold: int):
        """현재 fold를 설정하고 로그를 출력합니다."""
        self.current_fold = fold
        print(f"현재 Fold: {self.current_fold + 1}/{self.n_splits}")
        train_idx, val_idx = list(self.kfold.split(self.dataset))[self.current_fold]
        self.train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_idx)

    def next_fold(self):
        """Move to the next fold."""
        if self.current_fold + 1 < self.n_splits:
            self.set_fold(self.current_fold + 1)
        else:
            raise ValueError("All folds have been completed.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.data_config["data"]["batch_size"],
            num_workers=self.data_config["data"]["num_workers"],
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.data_config["data"]["batch_size"],
            num_workers=self.data_config["data"]["num_workers"],
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.data_config["data"]["batch_size"],
            num_workers=self.data_config["data"]["num_workers"],
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def _get_augmentation_transforms(self):
        """Returns augmentation transforms."""
        transform_list = [transforms.ToTensor()]
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_class = getattr(transforms, transform_config["name"])
            transform_list.append(transform_class(**transform_config["params"]))
        return transforms.Compose(transform_list)