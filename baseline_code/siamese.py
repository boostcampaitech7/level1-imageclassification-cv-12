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
from sklearn.model_selection import StratifiedKFold
from torch.amp import autocast, GradScaler
import random

##################################### Dataset ###############################################
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
            target = self.targets[index]
            return image, target

class SiameseDataset(Dataset):
    def __init__(self, dataset: CustomDataset):
        self.dataset = dataset
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        # 첫 번째 이미지 및 라벨 선택
        img1, label1 = self.dataset[index]

        # Positive or Negative Pair 생성 (50% 확률로 결정)
        if random.random() < 0.5:
            # Positive Pair (같은 클래스)
            while True:
                idx2 = random.randint(0, len(self.dataset) - 1)
                img2, label2 = self.dataset[idx2]
                if label1 == label2:
                    break
            label = 1  # 같은 클래스
        else:
            # Negative Pair (다른 클래스)
            while True:
                idx2 = random.randint(0, len(self.dataset) - 1)
                img2, label2 = self.dataset[idx2]
                if label1 != label2:
                    break
            label = 0  # 다른 클래스

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

################################## augmentation function #####################################
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


################################ Transform ####################################
class AlbumentationsTransform:
    def __init__(self, aug ,is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
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
                        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
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
                        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
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
                        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
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
                        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
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
                        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
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

############################# SiameseNetwork ################################
class SiameseNetwork(nn.Module):
    def __init__(self, model):
        super(SiameseNetwork, self).__init__()
        # Timm 라이브러리의 CoatNet3 모델 불러오기
        self.coatnet = model
        # 마지막 레이어 제거 (필요한 경우)
        if hasattr(self.coatnet, 'fc'):  # 만약 fc 속성이 있을 경우
            self.coatnet.fc = nn.Identity()
        elif hasattr(self.coatnet, 'classifier'):  # classifier 속성이 있을 경우
            self.coatnet.classifier = nn.Identity()

    def forward(self, input1, input2):
        # 두 입력 이미지에 대해 CoatNet3를 사용하여 특징 벡터 추출
        output1 = self.coatnet(input1)
        output2 = self.coatnet(input2)
        return output1, output2

    def get_embedding(self, input_image):
        # 개별 이미지에 대한 임베딩 벡터 추출
        return self.coatnet(input_image)
    
def contrastive_loss(output1, output2, label, margin=1.0):
    # 유클리드 거리 계산
    euclidean_distance = F.pairwise_distance(output1, output2)

    # 손실 계산 최적화
    loss_pos = (1 - label) * torch.square(euclidean_distance)
    loss_neg = label * torch.square(torch.clamp(margin - euclidean_distance, min=0.0))

    # 두 손실 항목을 더해 평균
    loss = loss_pos + loss_neg
    return loss.mean()

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
############################# trainer #############################

# 학습 함수 정의
def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for (img1, img2, labels) in progress_bar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        # Forward pass with autocast for mixed precision
        optimizer.zero_grad()
        with autocast(device_type=str(device)):
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

# 평가 함수 정의
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for (img1, img2, labels) in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Forward pass with autocast for mixed precision
            with autocast(device_type=str(device)):
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, labels)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(val_loader)

# 학습 및 평가 실행
def main(Siamese_model, train_loader, val_loader, device):
    model = Siamese_model.to(device)
    criterion = contrastive_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # Epoch 수 설정
    num_epochs = 10

    # 학습 루프
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


def inference(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    anchor_loader: DataLoader,          # 앵커 이미지와 라벨을 제공하는 DataLoader
    threshold: float = 1.0              # 유사성을 판단할 임계값
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()

    predictions = []

    # 앵커 이미지와 라벨을 미리 로드하여 처리
    all_anchor_images = []
    all_anchor_labels = []
    
    with torch.no_grad():
        for anchor_images, anchor_labels in anchor_loader:
            all_anchor_images.append(anchor_images)
            all_anchor_labels.extend(anchor_labels.cpu().numpy())

    # 모든 앵커 이미지를 하나의 텐서로 병합
    all_anchor_images = torch.cat(all_anchor_images).to(device)

    all_anchor_embeddings = []
    anchor_batch_size = 10  # 예시로 10개씩 처리

    with torch.no_grad():
        for i in range(0, all_anchor_images.size(0), anchor_batch_size):
            anchor_batch = all_anchor_images[i:i+anchor_batch_size]
            all_anchor_embeddings.append(model.get_embedding(anchor_batch))

    all_anchor_embeddings = torch.cat(all_anchor_embeddings)

    # 테스트 이미지에 대해 유사성 비교 시작
    progress_bar = tqdm(test_loader, desc="Inference", leave=False)
    
    with torch.no_grad():
        for test_images in progress_bar:
            test_images = test_images.to(device)
            best_labels_batch = []

            # 테스트 이미지의 임베딩을 미리 계산
            test_embeddings = model.get_embedding(test_images)

            torch.cuda.empty_cache()

            # 유클리드 거리 계산을 벡터화하여 앵커 임베딩과 비교
            for test_embedding in test_embeddings:
                # 앵커와 테스트 임베딩 간의 유클리드 거리 계산
                distances = F.pairwise_distance(test_embedding.unsqueeze(0), all_anchor_embeddings)

                # 가장 작은 거리를 가진 앵커 선택
                min_distance, min_index = torch.min(distances, dim=0)

                # 임계값을 기준으로 예측
                if min_distance.item() < threshold:
                    best_labels_batch.append(all_anchor_labels[min_index.item()])
                else:
                    best_labels_batch.append(-1)  # 유사한 앵커가 없으면 -1

            predictions.extend(best_labels_batch)

    return predictions

############################### train ##############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

model_name = 'coatnet_3_rw_224'

pt_root = './parameter/coatnet_3_rw_224'
traindata_dir = "./data/sketch/train"
traindata_info_file = "./data/sketch/train.csv"
testdata_dir = "./data/sketch/test"
testdata_info_file = "./data/sketch/test.csv"

train_info = pd.read_csv(traindata_info_file)
test_info = pd.read_csv(testdata_info_file)

# 총 class의 수를 측정.
num_classes = len(train_info['target'].unique())

########## 데이터 셋 선언 ##########
train_df, val_df = train_test_split(
    train_info,
    test_size=0.2,
    stratify=train_info['target']
)

transform_selector = TransformSelector(
    transform_type = "albumentations"
)

val_transform = transform_selector.get_transform(0, is_train=False) # 원본

train_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=train_df,
    transform=val_transform,
    num_classes=num_classes
)
val_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=val_df,
    transform=val_transform,
    num_classes=num_classes
)

# SiameseDataset 생성
siamese_train_dataset = SiameseDataset(train_dataset)

siamese_val_dataset = SiameseDataset(val_dataset)

train_loader = DataLoader(
    siamese_train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    siamese_val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Total training data size (after augmentation): {len(train_dataset)}")

########## 모델 선언 ###########
model_selector = ModelSelector(
    model_type='timm',
    num_classes=num_classes,
    model_name=model_name,
    pretrained=False
)
model = model_selector.get_model()

model.load_state_dict(
    torch.load(
        os.path.join(pt_root, f"best_model.pt"),
        map_location=device
    )
)

print("model parameter : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# Siamese 모델 선언
Siamese_model = SiameseNetwork(model = model)

print("Siamese_model parameter : ", sum(p.numel() for p in Siamese_model.parameters() if p.requires_grad))

########################## 학습 ##############################
main(Siamese_model, train_loader, val_loader, device)


########################## 추론 ##############################
test_transform = transform_selector.get_transform(0, is_train=False)

# 추론에 사용할 Dataset을 선언.
test_dataset = CustomDataset(
    root_dir=testdata_dir,
    info_df=test_info,
    transform=test_transform,
    num_classes=num_classes,
    is_inference=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    drop_last=False
)

anchor_df = train_info.groupby('target').first().reset_index()

anchor_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=anchor_df,
    transform=test_transform,
    num_classes=num_classes
)
anchor_loader = DataLoader(
    anchor_dataset,
    batch_size=8,
    shuffle=False,
    drop_last=False
)

predictions = inference(Siamese_model, device, test_loader, anchor_loader)

# 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
test_info['target'] = predictions
test_info = test_info.reset_index().rename(columns={"index": "ID"})
test_info

# DataFrame 저장
test_info.to_csv(f"./output_Siamese_{model_name}.csv", index=False)