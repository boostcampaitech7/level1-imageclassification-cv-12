import argparse
import importlib
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import importlib
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime  # 시간 정보를 가져오기 위해 추가

def main(config_path, use_wandb=True):
    # Config 파일 로드
    config = OmegaConf.load(config_path)

    # 데이터 모듈 설정
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(importlib.import_module(data_module_path), data_module_class)
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)

    # Wandb Logger 설정
    logger = WandbLogger(project="SWEEP!", name="Sweep") if use_wandb else None
    data_module.setup()

    # 현재 시간 포맷 설정
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    use_kfold = config.get("use_kfold", False)  # K-fold 사용 여부 확인
    val_results = []

    if use_kfold:
        print("\nTraining with K-fold validation.")
        # k-fold 학습 실행
        for fold in range(data_module.n_splits):
            print(f"\nFold {fold + 1}/{data_module.n_splits}")
            data_module.set_fold(fold)  # 현재 폴드 설정

            # 모델 모듈 동적 임포트 및 초기화
            model_module_path, model_module_class = config.model_module.rsplit(".", 1)
            ModelModuleClass = getattr(importlib.import_module(model_module_path), model_module_class)
            model = ModelModuleClass(config)  # 폴드마다 모델 초기화

            # 콜백 설정 (fold별로 체크포인트 경로 다르게 설정)
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"/data/ephemeral/home/JSM_Secret_Code/B3_simple_aug/fold_{fold + 1}",  # 폴드별 경로 지정
                filename=f"epoch={{epoch}}-val_loss={{val_loss:.2f}}-{current_time}",  # 파일명에 시간 추가
                monitor=config.callbacks.model_checkpoint.monitor,
                save_top_k=config.callbacks.model_checkpoint.save_top_k,
                mode=config.callbacks.model_checkpoint.mode,
            )
            early_stopping_callback = EarlyStopping(
                monitor=config.callbacks.early_stopping.monitor,
                patience=config.callbacks.early_stopping.patience,
                mode=config.callbacks.early_stopping.mode,
            )


            # checkpoint_path = "/data/ephemeral/home/JSM_Secret_Code/epoch_results_simpleAug/fold_1/epoch=epoch=25-val_loss=val_loss=0.98-20240922-172645.ckpt"


            # 트레이너 설정: 각 폴드마다 다른 로그 디렉토리 및 콜백 적용
            trainer = pl.Trainer(
                **config.trainer,
                callbacks=[checkpoint_callback, early_stopping_callback],
                logger=logger,
                default_root_dir=f"lightning_logs/fold_{fold + 1}",  # 각 폴드마다 다른 로그 디렉토리
            )


            # 훈련 시작
            trainer.fit(model, datamodule=data_module)

            # 각 fold에 대한 성능 평가 및 저장 (validation 관련 결과만 저장)
            val_accuracy = trainer.callback_metrics.get('val_accuracy', None)
            val_loss = trainer.callback_metrics.get('val_loss', None)

            if val_accuracy is not None and val_loss is not None:
                if isinstance(val_accuracy, torch.Tensor) and isinstance(val_loss, torch.Tensor):
                    print(f"Fold {fold + 1} - val_accuracy: {val_accuracy.item()}, val_loss: {val_loss.item()}")
                    val_results.append({
                        'val_accuracy': val_accuracy,
                        'val_loss': val_loss
                    })
        # 최종 val 성능 평균 계산
        if val_results:
            avg_val_accuracy = torch.mean(torch.stack([result['val_accuracy'] for result in val_results]))
            avg_val_loss = torch.mean(torch.stack([result['val_loss'] for result in val_results]))

            print(f"Final averaged validation accuracy: {avg_val_accuracy.item()}")
            print(f"Final averaged validation loss: {avg_val_loss.item()}")
        else:
            print("No valid results to average.")

    else:
        # 단일 데이터셋으로 학습 (K-fold가 아닌 경우)
        print("\nTraining without K-fold validation.")
        
        # 모델 모듈 동적 임포트 및 초기화
        model_module_path, model_module_class = config.model_module.rsplit(".", 1)
        ModelModuleClass = getattr(importlib.import_module(model_module_path), model_module_class)
        model = ModelModuleClass(config)  # 모델 초기화

        '''
        아래에서는 CheckPoint와 EarlyStopping을 정의한다.
        config 안에 들어있는 값들을 사용해서 다음과 같이 Object로 설정한 후 
        해당 값들을 pl.Trainer에서 사용하게 된다.
        
        '''
        # 콜백 설정
        checkpoint_callback = ModelCheckpoint(
            dirpath="/data/ephemeral/home/JSM_Secret_Code/epoch_results/CoAtNetB3_simpleBlur_RGB_Norm",  
            filename=f"epoch={{epoch}}-val_loss={{val_loss:.2f}}-{current_time}",  
            monitor=config.callbacks.model_checkpoint.monitor,
            save_top_k=config.callbacks.model_checkpoint.save_top_k,
            mode=config.callbacks.model_checkpoint.mode,
        )
        early_stopping_callback = EarlyStopping(
            monitor=config.callbacks.early_stopping.monitor,
            patience=config.callbacks.early_stopping.patience,
            mode=config.callbacks.early_stopping.mode,
        )

        '''
        logger 의 경우 위에서 wandb logger를 설정하여 wandb 설정을 하게 된다.
        이후 결과를 저장하는 곳을 default_root_dir 로 설정이 가능하다.
        비록 정의하지 않아도 되지만 현재 k-fold와 분류하기 위해서 다음과 같이 정의하였다.
        ''' 

        # checkpoint_path = "/data/ephemeral/home/JSM_Secret_Code/epoch_results_simpleAug/fold_1/epoch=epoch=25-val_loss=val_loss=0.98-20240922-172645.ckpt"


        # 트레이너 설정
        trainer = pl.Trainer(
            **config.trainer,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=logger,
            default_root_dir="lightning_logs/single",
        )


        '''
        위에서 setup 함수를 사용한 data_module을 사용
        즉, 모델과 해당 data_module을 사용한다.
        '''
        # 훈련 시작
        
        # trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
        trainer.fit(model, datamodule=data_module)

# 실행 : python train.py --config /path/to/train_config.yaml --use_wandb(option)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()
    main(args.config, args.use_wandb)