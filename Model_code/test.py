import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data.custom_data_module.data_module import DataModule
from src.plmodules.base_module import BaseModule



def main(config_path, checkpoint_path=None):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)  # 시드 값을 설정 파일에서 읽어오거나 기본값 42 사용
    data_module = DataModule(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # 체크포인트 경로 설정
    if checkpoint_path is None:
        if "checkpoint_path" in config:
            checkpoint_path = config.checkpoint_path
        else:
            raise ValueError("Checkpoint path is missing in the configuration file")

    print(f"Using checkpoint path: {checkpoint_path}")  # 경로가 제대로 설정되었는지 확인
    
    # 모델 설정
    model = BaseModule.load_from_checkpoint(checkpoint_path, config=config)

    # 트레이너 설정
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator, devices=config.trainer.devices
    )

    # 평가 시작
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with PyTorch Lightning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Path to the model checkpoint"
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint)
