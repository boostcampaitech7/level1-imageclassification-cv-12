import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomSketchDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        Custom dataset for loading sketch images and labels.
        
        Args:
            data_dir (str): Root directory of the images and CSV file.
            train (bool): If True, loads training data.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        """
        self.data_dir = data_dir  # 이미지 데이터가 저장된 루트 디렉토리 경로
        self.csv_path = os.path.join(data_dir, 'train.csv')  # 이미지 경로 및 레이블이 포함된 CSV 파일 경로
        self.train = train  # 학습 데이터인지 여부
        self.transform = transform  # 이미지 변환(transform) 설정

        # CSV 파일을 읽어 데이터프레임으로 저장
        self.labels_df = pd.read_csv(self.csv_path)

    def __len__(self):
        # 데이터셋의 총 이미지 개수 반환
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플(이미지와 라벨)을 반환합니다.
        
        Args:
            idx (int): 데이터의 인덱스
        Returns:
            tuple: (이미지, 라벨) 
        """
        # CSV 파일에서 클래스 이름, 이미지 경로, 레이블을 가져옵니다.
        class_name = self.labels_df.iloc[idx, 0]  # 클래스 이름
        img_path_in_csv = self.labels_df.iloc[idx, 1]  # 이미지 경로
        label = int(self.labels_df.iloc[idx, 2])  # 타겟 레이블

        # 이미지의 전체 경로를 구성합니다. /train 폴더를 경로에 추가합니다.
        img_path = os.path.join(self.data_dir, 'train', img_path_in_csv)

        # 이미지를 로드합니다.
        image = Image.open(img_path).convert("RGB")

        # 지정된 transform이 있다면 적용합니다.
        if self.transform:
            image = self.transform(image)

        # 이미지와 레이블을 반환합니다.
        return image, label