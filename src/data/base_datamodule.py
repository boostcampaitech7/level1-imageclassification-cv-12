# src/data/base_datamodule.py
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(pl.LightningDataModule):
    # 모든 값들을 초기화 하고 이는 setup 함수에서 정의된다고 합니다.

    # 여기서 config는 우리가 train, predcit, test중 원하는걸 선택하여 넣어줄수 있다.
    # 그리고 여기서 예를들어서 train/mnist_cnn_config.yaml 의 경로를 넣어준다면 CNN 기반으로 모델을 돌릴수 있다.


    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    # raise NotImpementedError를 통해서 setup이 정의되지 않으면 에러가 발생하도록 설정됩니다.
    # 인자로 stage를 받아 어느 단계인지 ( 학습, 검증, 테스트 ) 에 따라서 setup이 다르게 이뤄집니다. ( 사실 setup() 함수에서도 사용은 안하는듯 하다. )
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    # DataLoader에 미리 각각의 agumentation이 적용되어있다.
    # setup에서 정의한 모든 config들이 적용된다.
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            persistent_workers=True,
        )
