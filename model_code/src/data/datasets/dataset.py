import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

'''
    해당 함수는 실제 함수를 가져오은 함수입니다.
    data_config에서 data_dir를 통해서 실제 데이터, train_csv &  test, test_csv 을 가지고 오게 됩니다.

    Args : data_dir, train , transfrom

    Return : image, label
'''

class CustomDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.train = train
        self.data_dir = data_dir
        self.transform = transform
        self.labels = None

        # 학습용 데이터 처리
        if self.train:
            self.images_path = os.path.join(data_dir, 'augmented')
            self.labels_path = pd.read_csv(os.path.join(data_dir, 'train_augmented.csv'))


            if 'target' not in self.labels_path.columns:
                raise ValueError('현재 csv 파일에 target 이라는 column이 없어 에러를 발생시켰습니다.')

            self.labels = self.labels_path['target'].tolist()

        # 테스트용 데이터 처리
        else:
            self.images_path = os.path.join(data_dir, 'test')
            self.labels_path = pd.read_csv(os.path.join(data_dir, 'test.csv'))
            print(os.path.join(data_dir, 'test.csv'))

            if 'target' in self.labels_path.columns:
                raise ValueError('현재 test 데이터에 target (정답) 이 포함되어있습니다, 확인후 src/data/datasets/raw_data 33번째 줄을 수정해주세요')

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        img_path_csv = self.labels_path.loc[idx, 'image_path']
        img_path = os.path.join(self.images_path, img_path_csv)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 학습 데이터인 경우에만 라벨 반환
        label = int(self.labels[idx]) if self.train else -1

        return image, label