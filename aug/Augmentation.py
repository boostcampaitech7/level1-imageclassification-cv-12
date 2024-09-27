import os
import random
from PIL import Image, ImageFilter
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms.functional as F

'''

transformations에 원하는 증강을 추가

추가된 증강만큼 원본 데이터 수가 늘어나며

이후 입력받는 파일명으로 폴더 생성 후 증강된 이미지 저장

Return : 원본 + 증강된 이미지

'''


def augment_and_save_images(data_dir, save_dir, folder_name):
    images_path = os.path.join(data_dir, 'train')
    labels_path = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # 저장할 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    new_data = []

    # tqdm을 사용하여 진행 상황 표시
    for i, row in tqdm(labels_path.iterrows(), total=len(labels_path), desc="Processing images"):
        img_path = os.path.join(images_path, row['image_path'])

        # 경로가 존재하지 않는 경우, 경로를 수정
        if not os.path.exists(img_path):
            img_path = img_path.replace('original_', '')
            img_path = img_path.replace('/', '_')

        image = Image.open(img_path).convert("RGB")  # RGB 이미지로 변환
        label = row['target']

        # 원본 이미지 저장
        base_img_save_path = os.path.join(save_dir, f"{i}_original.jpg")
        image.save(base_img_save_path)
        new_data.append({"image_path": f"{i}_original.jpg", "target": label})

        # 증강 기법을 적용하여 저장
        transformations = [
            ('v_flipped', F.vflip(image)), # 원하는 증강 추가(v_flipped 는 예시)
        ]

        for aug_name, aug_image in transformations:
            aug_img_save_path = os.path.join(save_dir, f"{i}_{aug_name}.jpg")
            aug_image.save(aug_img_save_path)
            new_data.append({"image_path": f"{i}_{aug_name}.jpg", "target": label})

    # 새로운 데이터셋을 CSV 파일로 저장
    new_labels_path = os.path.join(save_dir, f'{folder_name}.csv')
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(new_labels_path, index=False)
    print(f"New augmented dataset saved at: {new_labels_path}")

# 사용자로부터 폴더명을 입력받는 과정 추가
folder_name = input("Enter the folder name for saving augmented data: ")

# 데이터 증강 적용 및 저장
augment_and_save_images(
    data_dir='/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch',
    save_dir=f'/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch/{folder_name}',
    folder_name=folder_name
)