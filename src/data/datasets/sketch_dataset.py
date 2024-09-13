import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class CustomSketchDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        Custom dataset for loading sketch images and labels.
        
        Args:
            data_dir (str): Root directory of the images and CSV file.
            train (bool): If True, loads training data.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        """
        self.train = train
        self.data_dir = data_dir
        self.transform = transform

        if self.train:
            self.images_path = os.path.join(data_dir, 'train')
            self.labels_path = pd.read_csv(os.path.join(data_dir, 'train.csv'))
            
            # 'target' 열이 존재하는지 확인
            if 'target' not in self.labels_path.columns:
                raise ValueError("train.csv 파일에 'target' 열이 없습니다.")
            
            self.labels = self.labels_path['target'].tolist()

        else:
            self.images_path = os.path.join(data_dir, 'test')
            self.labels_path = pd.read_csv(os.path.join(data_dir, 'test.csv'))
            
            # 'target' 열이 없어도 되는 경우를 위해 확인하지 않음
            if 'target' in self.labels_path.columns:
                self.labels = self.labels_path['target'].tolist()
            else:
                self.labels = None

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        # 이미지 경로와 레이블을 가져옵니다.
        img_path_in_csv = self.labels_path.loc[idx, 'image_path']  # 'image_path' 열로 접근
        img_path = os.path.join(self.images_path, img_path_in_csv)  # 전체 이미지 경로

        # 이미지 로드
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # 변환 적용
        if self.transform:
            image = self.transform(image)

        # 항상 튜플을 반환
        if self.train:
            label = int(self.labels_path.loc[idx, 'target'])  # 'target' 열로 접근
            return image, label
        else:
            return image, torch.tensor(-1)  # 테스트 데이터셋에서도 항상 두 개의 값을 반환