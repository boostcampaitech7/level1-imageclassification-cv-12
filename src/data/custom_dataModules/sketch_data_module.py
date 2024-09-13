import os
from typing import Optional

import torch
from torchvision import transforms

from src.data.base_datamodule import BaseDataModule
from src.data.collate_fns.sketch_collate_fn import sketch_collate_fn
from src.data.datasets.sketch_dataset import CustomSketchDataset
from src.utils.data_utils import load_yaml_config

# MNIST 데이터를 다운로드 받는다.
class SketchDataModule(BaseDataModule):
    def __init__(self, data_config_path: str, augmentation_config_path: str, seed: int):
        # 데이터 파일 경로 yaml 파일을 입력으로 받아 설정합니다. 
        self.data_config = load_yaml_config(data_config_path)

        # 데이터 증강과 관련된 yaml 파일을 경로로 받아 설정합니다.
        self.augmentation_config = load_yaml_config(augmentation_config_path)

        # SEED를 고정시키기 위한 함수 ( tran, test 시 일관성 유지를 위함 )
        self.seed = seed  # TODO

        # 부모 클래스 초기화 ( 해당 부모 클래스인 BaseDataModule의 경우 특정 데이터를 인자로 받도록 설계되어 있다. )
        super().__init__(self.data_config)

    def setup(self, stage: Optional[str] = None):
        # 시드 설정
        torch.manual_seed(self.seed)

        # 증강을 한다면 _get_augmentation_transforms 해당 함수에서 필요한 증강을 하나의 모듈로 가져오게된다.
        if self.augmentation_config["augmentation"]["use_augmentation"]:
            train_transforms = self._get_augmentation_transforms()
        else:
            train_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224))]
            )

        test_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224))]
        )

        # data/sketch 를 로드한다
        raw_data_path = self.config["data"]["raw_data_path"]

        # 실제 데이터를 가지고 와서 Train, Test로 나누어준다.
        self.train_dataset = CustomSketchDataset(
            data_dir=raw_data_path, train=True, transform=train_transforms
        )

        self.test_dataset = CustomSketchDataset(
            data_dir=raw_data_path, train=False, transform=test_transforms
        )

        # Split train dataset into train and validation
        train_size = int(
            len(self.train_dataset) * self.config["data"]["train_val_split"]
        )
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )

    # train 데이터를 로드 하는 함수이다.
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"], # 배치 사이즈 결정
            num_workers=self.config["data"]["num_workers"], # GPU 사용 
            shuffle=True,   # 섞을지 말지 결정
            collate_fn=sketch_collate_fn, # 어떻게 배치를 나눌지 결정하는 함수가 정의 됨
            persistent_workers=True, # 에폭이 끝나고 프로세스를 초기화 할지? 기본은 False인데 True로 설정하는것이 효율적이라고 한다
        )
    # val 데이터를 로드하는 함수이다.
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=sketch_collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=sketch_collate_fn,
            persistent_workers=True,
        )

    # augmentattion confing에 적용한 증강을 하나의 리스트에 담아서 리턴해준다.
    def _get_augmentation_transforms(self):
        transform_list = [
            transforms.ToTensor(),
        ]
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_class = getattr(transforms, transform_config["name"])
            transform_list.append(transform_class(**transform_config["params"]))
        # 위에서 설정한 모든 정규화를 Compose를 통해서 하나로 처리하게된다.
        return transforms.Compose(transform_list)
