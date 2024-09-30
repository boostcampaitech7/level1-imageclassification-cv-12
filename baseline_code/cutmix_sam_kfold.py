import os
from typing import Tuple, Any, Callable, List, Optional, Union
import cv2
import timm
import torch
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from torchcontrib.optim import SWA
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold
import random

"""
SAM (Sharpness-Aware Minimization), CutMix, KFold 교차 검증을 적용한 코드입니다.

1. KFold 교차 검증: 
   - StratifiedKFold를 사용하여 훈련 데이터셋을 여러 개의 폴드로 나누고, 각 폴드에 대해 모델을 훈련하고 검증합니다.
   s
2. SAM (Sharpness-Aware Minimization):
   - 모델의 손실을 최소화하는 데 도움이 되는 기법으로, 모델의 가중치를 두 단계로 업데이트하여 더욱 견고한 학습을 수행합니다.

3. CutMix:
   - 데이터 증강 기법으로, 두 이미지를 합성하여 새로운 이미지를 생성합니다. 이는 모델이 다양한 변형에 대해 학습하도록 돕습니다.
   - `rand_bbox`: 주어진 이미지 크기와 lambda 값에 따라 잘라낼 영역을 생성합니다.

 실행 방법 : python cutmix_sam_kfold.py 
 
"""

class CustomDataset(Dataset):
    def __init__(self, root_dir: str, info_df: pd.DataFrame, transform: Callable, num_classes : int ,is_inference: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.num_classes = num_classes
        self.image_paths = info_df['image_path'].tolist()

        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Albumentations 변환 적용
        if self.transform:
            augmented = self.transform(image=image)

            # 변환 결과가 딕셔너리일 경우 이미지만 가져옴
            if isinstance(augmented, dict):
                image = augmented['image']
            else:
                image = augmented  # 텐서로 바로 반환된 경우

        if self.is_inference:
            return image
        else:
            # One-hot encoding 후 float 타입으로 변환
            target = self.targets[index]
            return image, target


# Custom padding and resizing function with extra arguments handling
def resize_224x224(img,**kwargs):
    # 패딩할 상하/좌우 픽셀 계산
    ud, lr = (0, 0), (0, 0)

    # 이미지의 높이와 너비 중 더 큰 값을 찾음
    M = max(img.shape[:2])

    # 높이와 너비의 차이 계산
    s = img.shape[0] - img.shape[1]

    # 높이가 너비보다 긴 경우, 좌우 패딩
    if M == img.shape[0]:
        lr = (s // 2, s // 2 + s % 2)  # 홀수의 경우 한쪽에 1 추가
    # 너비가 높이보다 길거나 같은 경우, 상하 패딩
    else:
        ud = (-s // 2, -s // 2 + s % 2)

    # 패딩 적용 (배경을 흰색으로 설정, 즉 255)
    padded_img = np.pad(img, (ud, lr, (0, 0)), mode='constant', constant_values=255)

    # 224x224 크기로 리사이즈
    resized_img = cv2.resize(padded_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    return resized_img

class AlbumentationsTransform:
    def __init__(self, aug ,is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.ToGray(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        if is_train:
            if aug == 1:
                self.transform = A.Compose([
                    A.OneOf([
                        A.HorizontalFlip(p=1), 
                        A.VerticalFlip(p=1), 
                        A.Rotate(limit=90, p=1, border_mode=0, value=(255, 255, 255)),  # 최대 90도 회전
                    ], p=1.0),
                    A.OneOf([
                        A.Lambda(image=resize_224x224, p=1.0),
                        A.RandomResizedCrop(height=224, width=224, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
                    ], p=1.0),
                    *common_transforms
                ])
            elif aug == 2:
                self.transform = A.Compose([
                    A.OneOf([
                        A.Sharpen(p=1),  # Sharpen
                        A.Blur(blur_limit=9 ,p=1),  # Blur
                        A.CLAHE(clip_limit=4.0, p=1.0),
                        A.Compose([
                            A.InvertImg(p=1.0)
                        ], p=1.0)
                    ], p=1.0),
                    A.OneOf([
                        A.Lambda(image=resize_224x224, p=1.0),
                        A.RandomResizedCrop(height=224, width=224, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
                    ], p=1.0),
                    *common_transforms
                ])
            elif aug == 3:
                self.transform = A.Compose([
                    A.OneOf([
                        A.Affine(scale=1.0, translate_percent=(0.1, 0.1), rotate=15, shear=0.2, p=1.0),
                        A.ElasticTransform(alpha=1.0, sigma=50, p=1.0),
                        A.GridDistortion(p=1.0),  # Add GridDistortion
                    ], p=1.0),
                    A.OneOf([
                        A.Lambda(image=resize_224x224, p=1.0),
                        A.RandomResizedCrop(height=224, width=224, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
                    ], p=1.0),
                    *common_transforms
                ])
            elif aug == 4:
                self.transform = A.Compose([
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),  # Add Gaussian Noise
                        A.InvertImg(p=0.2)
                    ], p=1.0),
                    A.OneOf([
                        A.Lambda(image=resize_224x224, p=1.0),
                        A.RandomResizedCrop(height=224, width=224, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
                    ], p=1.0),
                    *common_transforms
                ])
            elif aug == 5:
                self.transform = A.Compose([
                    A.OneOf([
                        A.RandomBrightnessContrast(brightness_limit=(0.5, 1.3), contrast_limit=0.2, p=1.0),
                    ], p=1.0),
                    A.OneOf([
                        A.Lambda(image=resize_224x224, p=1.0),
                        A.RandomResizedCrop(height=224, width=224, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
                    ], p=1.0),
                    *common_transforms
                ])
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose([
                A.OneOf([
                    A.Lambda(image=resize_224x224, p=1.0),
                    #A.Resize(height=224, width=224, p=1),
                ], p=1.0),
                *common_transforms
            ])

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        #print(transformed['image'].shape)

        return transformed['image']  # 변환된 이미지의 텐서를 반환

class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type

        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, aug, is_train: bool):

        transform = AlbumentationsTransform(aug,is_train=is_train)

        return transform

"""# Model Class"""
class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self,
        model_type: str,
        num_classes: int,
        **kwargs
    ):
        if model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)

        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model

"""# Loss Class"""

class Loss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        return self.loss_fn(outputs, targets)

# CutMix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
"""# Trainer Class"""

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, loss_fn: torch.nn.modules.loss._Loss,
                 epochs: int, result_path: str, beta: float, cutmix: float, patient: float, SAM: bool, fold: int):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models = []
        self.lowest_loss = float('inf')
        self.cutmix = cutmix
        self.beta = beta
        self.patient = patient
        self.SAM = SAM
        self.fold = fold

    def save_model(self, epoch, loss):
        os.makedirs(self.result_path, exist_ok=True)
        current_model_path = os.path.join(self.result_path, f'model_epoch_{self.fold}_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, f'best_model_{self.fold}.pt')
            torch.save(self.model.state_dict(), best_model_path)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            r = np.random.rand(1)

            self.optimizer.zero_grad()

            if self.beta > 0 and self.cutmix > r:
                lam = np.random.beta(self.beta, self.beta)
                rand_index = torch.randperm(images.size()[0]).to(self.device)
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                
                if self.SAM:
                    outputs = self.model(images)
                    loss = loss = self.loss_fn(outputs, target_a) * lam + self.loss_fn(outputs, target_b) * (1. - lam)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.first_step(zero_grad=True)

                    outputs = self.model(images)
                    loss_second = loss = self.loss_fn(outputs, target_a) * lam + self.loss_fn(outputs, target_b) * (1. - lam)

                    loss_second.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.second_step(zero_grad=True)

                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())
                else:
                    outputs = self.model(images)
                    loss = loss = self.loss_fn(outputs, target_a) * lam + self.loss_fn(outputs, target_b) * (1. - lam)

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())
            else:
                if self.SAM:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, targets)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.first_step(zero_grad=True)

                    outputs = self.model(images)
                    loss_second = self.loss_fn(outputs, targets)

                    loss_second.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.second_step(zero_grad=True)

                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())
                else:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, targets)

                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)

                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                progress_bar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples * 100)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples * 100
        return avg_loss, accuracy

    def train(self):
        best_val_accuracy = 0.0
        count = 0
        for epoch in range(self.epochs):
            if count >= self.patient:
                print(f"early stopping!! epoch : {epoch+1}")
                break
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            if val_accuracy > best_val_accuracy:
                count = 0
                best_val_accuracy = val_accuracy
            else:
                count += 1
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                  f"Validation accuracy: {val_accuracy:.4f}, patient: {count}\n")
            self.save_model(epoch, val_loss)
            self.scheduler.step()
        
        return best_val_accuracy

### https://github.com/davda54/sam

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
# 모델 추론을 위한 함수
def inference(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)

            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가

    return predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'coatnet_3_rw_224'
traindata_dir = "./data/sketch/train"
traindata_info_file = "./data/sketch/train.csv"
save_result_path = f"./parameter/{model_name}"
save_result_csv = f"./csv/{model_name}"

if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

if not os.path.exists(save_result_csv):
    os.makedirs(save_result_csv)

# 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
train_info = pd.read_csv(traindata_info_file)

# 총 class의 수를 측정.
num_classes = len(train_info['target'].unique())

# StratifiedKFold 설정
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

acc_list = []
# StratifiedKFold를 사용하여 train/val 나누기
for fold, (train_idx, val_idx) in enumerate(skf.split(train_info, train_info['target'])):
    print(f"Fold {fold + 1}")
    
    # train과 validation 데이터프레임 나누기
    train_df = train_info.iloc[train_idx]
    val_df = train_info.iloc[val_idx]

    print(f"Train size: {train_df.shape}, Validation size: {val_df.shape}")

    # 학습에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = "albumentations"
    )
    #train_transform = transform_selector.get_transform(is_train=True) # augmentation
    val_transform = transform_selector.get_transform(0, is_train=False) # 원본

    # 학습에 사용할 Dataset을 선언.
    train_dataset_1 = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=transform_selector.get_transform(1, is_train=True),
        num_classes=num_classes
    )
    train_dataset_2 = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=transform_selector.get_transform(2, is_train=True),
        num_classes=num_classes
    )
    train_dataset_3 = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=transform_selector.get_transform(3, is_train=True),
        num_classes=num_classes
    )
    train_dataset_4 = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=transform_selector.get_transform(4, is_train=True),
        num_classes=num_classes
    )
    train_dataset_5 = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=transform_selector.get_transform(5, is_train=True),
        num_classes=num_classes
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform,
        num_classes=num_classes
    )
    augmented_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=val_transform,
        num_classes=num_classes
    )
    combined_train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5, augmented_dataset])
    # 학습에 사용할 DataLoader를 선언.
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    print(f"Total training data size (after augmentation): {len(combined_train_dataset)}")

    # 학습에 사용할 Model을 선언.
    model_selector = ModelSelector(
        model_type='timm',
        num_classes=num_classes,
        model_name=model_name,
        pretrained=True
    )
    model = model_selector.get_model()
    #model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    print("model parameter : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # 선언된 모델을 학습에 사용할 장비로 셋팅.
    model = model.to(device)

    base_optimizer = torch.optim.SGD

    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

    # 학습에 사용할 Loss를 선언.
    loss_fn = Loss()

    # 앞서 선언한 필요 class와 변수들을 조합해, 학습을 진행할 Trainer를 선언.
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=20,
        result_path=save_result_path,
        beta=1.0,
        cutmix=0.5,
        patient=5,
        SAM=True,
        fold=fold
    )

    # 모델 학습.
    acc = trainer.train()
    print(f"fold {fold} final validation accuracy : {acc}")
    acc_list.append(acc)

    """# Inference"""

    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    testdata_dir = "./data/sketch/test"
    testdata_info_file = "./data/sketch/test.csv"

    # 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    test_info = pd.read_csv(testdata_info_file)

    # 총 class 수.
    num_classes = 500

    # 추론에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = "albumentations"
    )
    test_transform = transform_selector.get_transform(0, is_train=False)

    # 추론에 사용할 Dataset을 선언.
    test_dataset = CustomDataset(
        root_dir=testdata_dir,
        info_df=test_info,
        transform=test_transform,
        num_classes=num_classes,
        is_inference=True
    )

    # 추론에 사용할 DataLoader를 선언.
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False
    )

    # 추론에 사용할 장비를 선택.
    # torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 추론에 사용할 Model을 선언.
    model_selector = ModelSelector(
        model_type='timm',
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False
    )
    model = model_selector.get_model()

    #model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    # 선언된 모델을 학습에 사용할 장비로 셋팅.
    model = model.to(device)

    # best epoch 모델을 불러오기.
    model.load_state_dict(
        torch.load(
            os.path.join(save_result_path, f"best_model_{fold}.pt"),
            map_location='cpu'
        )
    )
    # predictions를 CSV에 저장할 때 형식을 맞춰서 저장
    # 테스트 함수 호출
    predictions = inference(
        model=model,
        device=device,
        test_loader=test_loader
    )

    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info

    # DataFrame 저장
    test_info.to_csv(f"{save_result_csv}/kfold_{fold}_{acc}.csv", index=False)


print(f"kfold mean accuracy : {np.mean(np.array(acc_list))}")
best_fold = np.argmax(acc_list) + 1
print(f"kfold best fold : {best_fold}")