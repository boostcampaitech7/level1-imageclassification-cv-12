import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from src.models.base_model import BaseModel


class SketchBaseModule(pl.LightningModule):
    def __init__(self, config):
        super(SketchBaseModule, self).__init__()
        self.config = config
        self.model = BaseModel(config.model)

        # 수정: num_classes를 500으로 설정
        self.precision = MulticlassPrecision(num_classes=500, average="macro")
        self.recall = MulticlassRecall(num_classes=500, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=500, average="macro")
        
        # wandb logger 설정 (필요시 프로젝트명과 이름 변경)
        self.wandb_logger = WandbLogger(project="SketchProject", name="Sketch_Test")
        
        self.test_results = {}
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        # y_hat을 클래스 레이블로 변환
        preds = torch.argmax(y_hat, dim=1)
        
        # 로깅
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_precision", self.precision(preds, y), on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_recall", self.recall(preds, y), on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train_f1_score", self.f1_score(preds, y), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        # y_hat을 클래스 레이블로 변환
        preds = torch.argmax(y_hat, dim=1)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_precision", self.precision(preds, y), prog_bar=True)
        # self.log("val_recall", self.recall(preds, y), prog_bar=True)
        # self.log("val_f1_score", self.f1_score(preds, y), prog_bar=True)
        return loss

    # 에포크 시작시 output을 초기화 시킨다.
    def on_test_epoch_start(self):
        self.test_step_outputs = []  # 테스트 에포크 시작 시 초기화

    # 한번의 테스트 (배치 ) 에 대한 처리를 진행한다.
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        # y_hat을 클래스 레이블로 변환
        preds = torch.argmax(y_hat, dim=1)

        self.log("test_loss", loss, prog_bar=True)
        # self.log("test_precision", self.precision(preds, y), prog_bar=True)
        # self.log("test_recall", self.recall(preds, y), prog_bar=True)
        # self.log("test_f1_score", self.f1_score(preds, y), prog_bar=True)
        output = {"loss": loss, "preds": preds, "targets": y}
        self.test_step_outputs.append(output)
        return output

    # 모든 테스트가 종료되었을때 호출된다 ( acc를 계산하는 과정 )
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        preds = torch.cat([output["preds"] for output in outputs])
        targets = torch.cat([output["targets"] for output in outputs])

        self.test_results["predictions"] = preds
        self.test_results["targets"] = targets

        accuracy = (preds == targets).float().mean()
        self.log("test_accuracy", accuracy, prog_bar=True)

        self.test_step_outputs.clear()  # 메모리 정리

    # 예측
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    # optimizer를 가지고 온다.
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