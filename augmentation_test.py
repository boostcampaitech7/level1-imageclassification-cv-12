import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2

def apply_sobel_filter(image):
    # Sobel 필터를 적용하기 위해 이미지를 numpy 배열로 변환
    image_np = np.array(image)
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Sobel 필터 적용
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)

    # 결과를 uint8 형식으로 변환하여 RGB로 다시 변환
    sobel = np.uint8(np.absolute(sobel))
    sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(sobel_rgb)

def augment_and_save_images(data_dir, save_dir, num_augmentations):
    # 원본 이미지를 불러옵니다.
    images_path = os.path.join(data_dir, 'train')
    labels_path = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # 증강 변환 설정 (색상 반전, 회전, 그레이스케일, 추가 회전, Sobel 필터)
    augmentation_transforms = [
        transforms.RandomRotation(degrees=15),  # 회전
        transforms.RandomInvert(p=1.0),  # 색상 반전
        transforms.Grayscale(num_output_channels=3),  # 그레이스케일 변환 후 RGB 채널 유지
        transforms.RandomRotation(degrees=30),  # 추가 회전
        apply_sobel_filter  # Sobel 필터 적용
    ]
    
    # 저장할 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    new_data = []

    # tqdm을 사용하여 진행 상황 표시
    for i, row in tqdm(labels_path.iterrows(), total=len(labels_path), desc="Processing images"):
        img_path = os.path.join(images_path, row['image_path'])
        image = Image.open(img_path).convert("RGB")
        label = row['target']
        
        # 원본 이미지 저장
        base_img_save_path = os.path.join(save_dir, f"{i}_original.jpg")
        image.save(base_img_save_path)
        new_data.append({"image_path": f"{i}_original.jpg", "target": label})
        
        # 증강된 이미지 생성 및 저장
        for j in range(num_augmentations):
            # Sobel 필터의 경우 별도로 처리
            if augmentation_transforms[j] == apply_sobel_filter:
                augmented_image = apply_sobel_filter(image)
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomChoice(augmentation_transforms[:-1]),  # Sobel 필터 제외 랜덤한 증강 적용
                    transforms.ToPILImage()
                ])
                augmented_image = transform(image)

            aug_img_save_path = os.path.join(save_dir, f"{i}_aug_{j}.jpg")
            augmented_image.save(aug_img_save_path)
            new_data.append({"image_path": f"{i}_aug_{j}.jpg", "target": label})

    # 새로운 데이터셋을 CSV 파일로 저장
    new_labels_path = os.path.join(save_dir, 'train_augmented.csv')
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(new_labels_path, index=False)
    print(f"New augmented dataset saved at: {new_labels_path}")

# 사용자로부터 폴더명을 입력받는 과정 추가
folder_name = input("Enter the folder name for saving augmented data: ")

# 데이터 증강 적용 및 저장
augment_and_save_images(
    data_dir='/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch',  # 올바른 데이터셋 경로
    save_dir=f'/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch/{folder_name}',  # 입력받은 폴더명 사용
    num_augmentations=5  # 각 이미지 당 증강 개수
)