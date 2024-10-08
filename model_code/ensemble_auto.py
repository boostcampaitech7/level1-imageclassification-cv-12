import argparse
import importlib
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import pytz
from datetime import datetime
import re  # 정규 표현식 사용을 위한 import


'''
    해당 파일은 앙상블을 수행하는 코드입니다.
    해당 코드는 ckpt 적용시 작성되는 loss를 활용하여 loss의 역수를 곱하여 자동으로 가중치를 설정하게 됩니다.

    실행 방법  : python ensemble_auto.py --config /path/to/ensemble.yaml
'''

def extract_val_loss_from_checkpoint(checkpoint_path):
    # 체크포인트 파일명에서 val_loss 값을 추출하는 정규식
    match = re.search(r"val_loss=([\d\.]+)", checkpoint_path)
    if match:
        return float(match.group(1))  # 추출된 val_loss 값을 float으로 변환하여 반환
    else:
        return None  # val_loss 값을 찾지 못한 경우

def main(config_path, use_wandb=False):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 동적 임포트
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 14)
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # 모델 모듈 동적 임포트 및 설정
    model_modules = []
    model_val_losses = []
    
    for model_info in config.models:
        model_module_path, model_module_class = model_info.module.rsplit(".", 1)
        ModelModuleClass = getattr(
            importlib.import_module(model_module_path), model_module_class
        )
        model = ModelModuleClass.load_from_checkpoint(
            model_info.checkpoint, config=config
        )
        model_modules.append(model)

        # 체크포인트 파일명에서 val_loss 추출
        val_loss = extract_val_loss_from_checkpoint(model_info.checkpoint)
        if val_loss is None:
            print(f"Warning: No val_loss found in the checkpoint file name for model {type(model).__name__}")
            continue
        model_val_losses.append(val_loss)

    if not model_val_losses:
        print("Error: No valid models with val_loss found. Exiting.")
        return

    # 모델의 가중치 계산 (val_loss의 역수로 가중치 설정)
    model_weights = [1 / loss for loss in model_val_losses]
    total_weight = sum(model_weights)
    model_weights = [weight / total_weight for weight in model_weights]  # 정규화

    # 트레이너 설정
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator, devices=config.trainer.devices, logger=False, enable_progress_bar=False
    )

    # 예측 시작
    all_logits = []  # 각 모델의 예측 logits (확률 값)을 저장할 리스트
    result_dfs = []  # 각 모델의 결과를 저장할 리스트

    for model_idx, model in enumerate(model_modules):
        trainer.test(model, datamodule=data_module)
        logits = model.test_results.get("logits", None)  # logits 가져오기 (raw 예측값)

        # 예측 결과가 없을 경우 처리
        if logits is None:
            print(f"Error: No logits found for model {type(model).__name__}")
            continue

        logits_tensor = torch.tensor(logits)

        # 디버깅: 각 모델의 logits 출력 (5개의 샘플만)
        print(f"Model {model_idx} ({type(model).__name__}), Logits for 5 samples:")
        print(logits_tensor[:5])

        all_logits.append(logits_tensor)

        # 결과를 DataFrame으로 변환하고 저장
        result_df = model.test_df.copy()  # 각 모델의 DataFrame 복사
        result_df["target"] = torch.argmax(logits_tensor, dim=1).cpu().numpy()  # 예측값으로 target 설정
        result_dfs.append(result_df)

    if not all_logits:
        print("Error: No logits found across models. Exiting.")
        return

    # 가중치 적용된 앙상블 예측 (logits에 가중치를 적용한 후 평균)
    weighted_logits = sum(weight * logits for weight, logits in zip(model_weights, all_logits))
    ensemble_logits = weighted_logits / sum(model_weights)

    # 최종 예측을 argmax로 결정
    y_pred = ensemble_logits.argmax(dim=1).cpu().numpy()

    # 앙상블 예측을 5개 샘플에 대해 출력
    print(f"Ensemble Predictions for 5 samples: {y_pred[:5]}")

    # 실제 레이블 y_true를 가져오기
    y_true = []
    for batch in data_module.test_dataloader():
        _, targets = batch
        y_true.extend(targets.cpu().numpy())

    y_true = torch.tensor(y_true).numpy()  # numpy 배열로 변환

    # 성능 지표 계산
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Ensemble F1 Score: {f1}")
    print(f"Ensemble Precision: {precision}")
    print(f"Ensemble Recall: {recall}")

    # 앙상블 결과 저장
    kst = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(pytz.utc).astimezone(kst).strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"/data/ephemeral/home/JSM_Secret_Code/results_ensemble/ensemble_{current_time}.csv"

    # 가장 첫 번째 결과 DataFrame을 기반으로 ID와 image_path 유지
    ensemble_df = result_dfs[0][["ID", "image_path"]].copy()
    ensemble_df["target"] = y_pred  # 최종 앙상블 예측 값 설정
    ensemble_df.to_csv(output_path, index=False)
    print(f"Saved ensemble results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict with an ensemble of trained models using PyTorch Lightning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()
    main(args.config, args.use_wandb)