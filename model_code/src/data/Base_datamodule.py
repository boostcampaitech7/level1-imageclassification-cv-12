 # src/data/base_datamodule.py
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

'''
    해당 코드는 pytorch ligthening 에서 필요로 하는 기본적인 함수드를 구현해둔 코드입니다.

    Args : Config 


    Return : DataLoder 등 data_moudle에서 사용 가능하도록 return 해줍니다.

'''
class BaseDataModule(pl.LightningDataModule):


    def __init__(self, config : Dict[str, Any]) :
        super().__init__()
        self.train_dataset : Optional[Dataset] = None
        self.val_dataset : Optional[Dataset] = None
        self.test_dataset : Optional[Dataset] = None


    def setup(self, stage : Optional[str] = None):
        raise NotImplementedError


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
