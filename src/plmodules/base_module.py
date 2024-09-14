import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
)
from src.models.ResNet101 import ResNet101
import pandas as pd
import os
from datetime import datetime
from datetime import datetime
import pytz

class SketchBaseModule(pl.LightningModule):
    def __init__(self, config):
        super(SketchBaseModule, self).__init__()
        self.config = config
        self.model = ResNet101(config.model)

        # 수정: num_classes를 500으로 설정
        self.precision = MulticlassPrecision(num_classes=500, average="macro")
        self.recall = MulticlassRecall(num_classes=500, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=500, average="macro")

        # 추가: Validation Accuracy를 위한 Metric 초기화
        self.accuracy = MulticlassAccuracy(num_classes=500)
        
        # wandb logger 설정 (필요시 프로젝트명과 이름 변경)
        self.wandb_logger = WandbLogger(project="SketchProject", name="Sketch_Test")
        
        self.test_results = {} 
        self.test_step_outputs = []

        # 테스트용 CSV 파일 경로
        self.test_csv_path = "/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch/test.csv"  # 'test.csv' 파일 경로

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        preds = torch.argmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)

        # 수정: Validation Accuracy 계산 및 로그 추가
        accuracy = self.accuracy(preds, y)
        self.log("val_accuracy", accuracy * 100, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.test_step_outputs = []

        # 테스트 CSV 파일 읽기
        self.test_df = pd.read_csv(self.test_csv_path)

    def test_step(self, batch, batch_idx):
        # 배치가 여러 개의 값을 가질 수 있으므로, 길이를 확인하여 처리합니다.
        if isinstance(batch, tuple) or isinstance(batch, list):
            if len(batch) == 1:
                x = batch[0]  # 배치가 이미지 데이터만 가진 경우
            elif len(batch) == 2:
                x, _ = batch  # 배치가 이미지와 레이블을 가진 경우
            else:
                raise ValueError(f"Unexpected number of values in batch: {len(batch)}")  # 에러 메시지에 길이 포함
        else:
            raise ValueError("Batch should be a tuple or list.")

        y_hat = self.forward(x)
        preds = torch.argmax(y_hat, dim=1)
        
        # 예측 결과를 리스트에 추가합니다.
        self.test_step_outputs.append(preds)

        # 예측 결과의 평균을 로그로 남깁니다. (단일 스칼라 값)
        self.log("test_predictions_mean", preds.float().mean(), prog_bar=True)
        
        return preds

    # def on_test_epoch_end(self):
    #     # 모든 예측 결과를 결합
    #     if self.test_step_outputs:
    #         preds = torch.cat(self.test_step_outputs).cpu().numpy()

    #         # 모델 이름과 현재 날짜 및 시간으로 파일 이름 생성
    #         model_name = type(self.model).__name__
    #         current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #         output_path = f"/data/ephemeral/home/level1-imageclassification-cv-12/results/{model_name}_{current_time}_now.csv"

    #         # ID를 생성하고 결과를 데이터프레임으로 변환
    #         self.test_df["target"] = preds
    #         self.test_df.insert(0, 'ID', range(len(self.test_df)))  # ID 열 추가

    #         # 예측 결과를 CSV 파일에 추가
    #         self.test_df.to_csv(output_path, index=False)
    #         print(f"Saved output CSV to {output_path}")

    #     else:
    #         print("No test outputs to save.")

    #     # 결과 리스트 초기화
    #     self.test_step_outputs.clear()

    def on_test_epoch_end(self):
        # 모든 예측 결과를 결합
        if self.test_step_outputs:
            preds = torch.cat(self.test_step_outputs).cpu().numpy()

            # 예측한 값의 길이와 데이터프레임의 길이를 맞추기 위한 처리
            expected_length = len(self.test_df)
            if len(preds) > expected_length:
                preds = preds[:expected_length]  # 예측된 값의 길이를 데이터프레임의 길이에 맞춤
            elif len(preds) < expected_length:
                raise ValueError(f"Predictions length {len(preds)} does not match DataFrame length {expected_length}")

            # ID 열 추가
            self.test_df["ID"] = range(len(self.test_df))
            
            # 예측 결과를 "target" 열에 추가
            self.test_df["target"] = preds

            # 필요한 순서대로 열을 재정렬
            self.test_df = self.test_df[["ID", "image_path", "target"]]


            kst = pytz.timezone('Asia/Seoul')

            # 모델 이름과 현재 날짜 및 시간으로 파일 이름 생성
            model_name = type(self.model).__name__
            current_time = datetime.now(pytz.utc).astimezone(kst).strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"/data/ephemeral/home/level1-imageclassification-cv-12/results/{model_name}_{current_time}.csv"

            # 변경된 경로로 CSV 파일 저장
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
            scheduler_class = getattr(
                torch.optim.lr_scheduler, self.config.scheduler.name
            )
            scheduler = scheduler_class(optimizer, **self.config.scheduler.params)
            return [optimizer], [scheduler]
        else:
            return optimizer