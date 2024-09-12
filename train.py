import argparse
import importlib
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt




def main(config_path, use_wandb=False):
    
    # 우리가 학습하고 싶으면 모델의 config 파일을 --config PATH로 넘겨주는 config를 의미한다.
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 동적 임포트
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )

    # 여기서 데이터와 증강에 대한 정보를 가지고 온다 ( configs augmentatoin_confing, data_configs )
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path


    # 데이터를 가지고와서 증강까지 처리하여 setup()을 진행한다.
    seed = config.get("seed", 42)
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # 모델 모듈 동적 임포트
    model_module_path, model_module_class = config.model_module.rsplit(".", 1)
    ModelModuleClass = getattr(
        importlib.import_module(model_module_path), model_module_class
    )

    # 모델 설정
    model = ModelModuleClass(config)

    # Wandb 로거 설정
    logger = None
    if use_wandb:
        logger = WandbLogger(project="Sketch Image", name="BaseModel_test")

    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.model_checkpoint.monitor,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        mode=config.callbacks.model_checkpoint.mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
    )

    # 트레이너 설정
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
    )

    # 훈련 시작
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()
    main(args.config, args.use_wandb)