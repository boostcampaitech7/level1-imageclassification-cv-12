import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
)
from src.models.EfficientNetV2 import EfficientNetV2
import pandas as pd
import os
from datetime import datetime
import pytz
import numpy as np

# Progress Bar 수정
from pytorch_lightning.callbacks import TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    # 해당 경로에 각각 실험 결과들이 저장되게 된다.
    def __init__(self, log_file="training_log.txt"):
        super().__init__()
        self.log_file = log_file

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        print(f"\nEpoch {trainer.current_epoch + 1}/{trainer.max_epochs}")
    
    def on_train_epoch_end(self, unused=None):
        super().on_train_epoch_end()
        # When all epochs for the current fold are done, move to the next fold
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            print(f"Finished training fold {self.datamodule.current_fold + 1}/{self.datamodule.n_splits}")
            if self.datamodule.current_fold + 1 < self.datamodule.n_splits:
                self.datamodule.next_fold()
                print(f"Starting next fold: {self.datamodule.current_fold + 1}/{self.datamodule.n_splits}")
                self.trainer.reset_train_dataloader(self)
                self.trainer.reset_val_dataloader(self)

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        val_loss = trainer.callback_metrics['val_loss']
        val_accuracy = trainer.callback_metrics['val_accuracy'] * 100
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print("-" * 50)

        # 검증 에폭이 끝날 때마다 로그 파일에 기록
        with open(self.log_file, "a") as f:
            f.write(f"Epoch {trainer.current_epoch + 1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")

# 모델 모듈 수정
class BaseModule(pl.LightningModule):
    def __init__(self, config):
        super(BaseModule, self).__init__()
        self.config = config
        self.model = EfficientNetV2(config.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # CrossEntropy Loss with Label Smoothing 추가
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

        # Metrics 설정
        self.precision = MulticlassPrecision(num_classes=500, average="macro")
        self.recall = MulticlassRecall(num_classes=500, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=500, average="macro")
        self.accuracy = MulticlassAccuracy(num_classes=500)

        # Wandb Logger 추가
        self.wandb_logger = WandbLogger(project="SketchProject", name="Sketch_Test")
        
        self.test_results = {} 
        self.test_step_outputs = []

        # 테스트용 CSV 파일 경로
        self.test_csv_path = "/data/ephemeral/home/JSM_Secret_Code/data/sketch/data/test.csv"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", self.precision(y_hat, y), on_epoch=True)
        self.log("train_recall", self.recall(y_hat, y), on_epoch=True)
        self.log("train_f1_score", self.f1_score(y_hat, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)  # Using CrossEntropyLoss with label smoothing

        preds = torch.argmax(y_hat, dim=1)
        accuracy = self.accuracy(preds, y)
        self.log("val_accuracy", accuracy * 100, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_precision", self.precision(y_hat, y))
        self.log("val_recall", self.recall(y_hat, y))
        self.log("val_f1_score", self.f1_score(y_hat, y))

        return loss

    def on_test_epoch_start(self):
        self.test_step_outputs = []
        self.test_df = pd.read_csv(self.test_csv_path)

    
    def test_step(self, batch, batch_idx):
        if isinstance(batch, tuple) or isinstance(batch, list):
            x = batch[0] if len(batch) == 1 else batch[0]
        else:
            raise ValueError("Batch should be a tuple or list.")

        y_hat = self.forward(x)  # 모델의 raw logits 출력
        logits = F.softmax(y_hat, dim=-1)  # Logits를 확률로 변환 (softmax 적용)

        preds = torch.argmax(logits, dim=1)  # 예측값은 가장 높은 확률을 가진 클래스
        self.test_step_outputs.append(preds)

        # logits 저장
        if 'logits' not in self.test_results:
            self.test_results['logits'] = logits.cpu().numpy()  # Logits 저장
        else:
            self.test_results['logits'] = np.concatenate(
                (self.test_results['logits'], logits.cpu().numpy()), axis=0
            )

        # Logging
        self.log("test_predictions_mean", preds.float().mean(), prog_bar=True)
        
        return logits  # raw logits 반환




    def on_test_epoch_end(self):
        if self.test_step_outputs:
            preds = torch.cat(self.test_step_outputs).cpu().numpy()
            expected_length = len(self.test_df)
            if len(preds) > expected_length:
                preds = preds[:expected_length]
            elif len(preds) < expected_length:
                raise ValueError(f"Predictions length {len(preds)} does not match DataFrame length {expected_length}")

            self.test_df["ID"] = range(len(self.test_df))
            self.test_df["target"] = preds
            self.test_df = self.test_df[["ID", "image_path", "target"]]

            kst = pytz.timezone('Asia/Seoul')
            model_name = type(self.model).__name__
            current_time = datetime.now(pytz.utc).astimezone(kst).strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"/data/ephemeral/home/JSM_Secret_Code/results/{model_name}_{current_time}.csv"

            # 경로 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.test_df.to_csv(output_path, index=False)
            print(f"Saved output CSV to {output_path}")
        else:
            print("No test outputs to save.")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.config.optimizer.name)
        optimizer = optimizer_class(self.parameters(), **self.config.optimizer.params)

        if hasattr(self.config, "scheduler"):
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler.name)
            scheduler = scheduler_class(optimizer, **self.config.scheduler.params)
            return [optimizer], [scheduler]
        else:
            return optimizer


