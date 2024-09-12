import os
from typing import Optional

import torch
from torchvision import transforms

from src.data.base_datamodule import BaseDataModule
from src.data.collate_fns.mnist_collate_fn import mnist_collate_fn
from src.data.datasets.mnist_dataset import CustomMNISTDataset
from src.utils.data_utils import load_yaml_config


# MNIST 데이터를 다운로드 받는다.
class MNISTDataModule(BaseDataModule):
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

        # 증강할지 말지 결정한다. 기본으로 config에는 False로 설정되어있다.
        if self.augmentation_config["augmentation"]["use_augmentation"]:
            train_transforms = self._get_augmentation_transforms()

        # 증강하지 않는다면 기본 Nomalization 진행한다.
        else:
            train_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

        test_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Load datasets
        raw_data_path = self.config["data"]["raw_data_path"]

        # 실제 데이터를 가지고 와서 Train, Test로 나누어준다.
        self.train_dataset = CustomMNISTDataset(
            data_dir=raw_data_path, train=True, transform=train_transforms
        )

        self.test_dataset = CustomMNISTDataset(
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
            collate_fn=mnist_collate_fn, # 어떻게 배치를 나눌지 결정하는 함수가 정의 됨
            persistent_workers=True, # 에폭이 끝나고 프로세스를 초기화 할지? 기본은 False인데 True로 설정하는것이 효율적이라고 한다
        )
    # val 데이터를 로드하는 함수이다.
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=mnist_collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=mnist_collate_fn,
            persistent_workers=True,
        )

    def _get_augmentation_transforms(self):
        transform_list = [
            transforms.ToTensor(),
            # 평균, 표준편차 이렇게 설정한 이유는 해당 값이 MNIST 데이터의 평균이기 때문이다. ( 모든 데이터의 평균과 표준편차로 정규화 하는 것도 하나의 방법이라고 한다. )
            transforms.Normalize((0.1307,), (0.3081,)),
        ]


        '''
        # configs/augmentation_configs/mnist_augmentation.yaml
        augmentation:
        use_augmentation: false
        transforms:
            - name: RandomRotation
            params:
                degrees: 10
            - name: RandomAffine
            params:
                degrees: 0
                translate: [0.1, 0.1]
                scale: [0.9, 1.1]
            - name: Normalize
            params:
                mean: [0.1307]
                std: [0.3081]

        /configs/augmentation_configs 안에 위와 같이 설정되어있다. 그래서 name, params를 사용해서 다음과 같이 정규화가 가능한거다.
        '''
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_class = getattr(transforms, transform_config["name"])
            transform_list.append(transform_class(**transform_config["params"]))
        # 위에서 설정한 모든 정규화를 Compose를 통해서 하나로 처리하게된다.
        return transforms.Compose(transform_list)
