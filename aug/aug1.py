import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import torchvision.transforms.functional as F


def center_crop_80_percent(image):
    # 이미지의 원래 크기
    original_width, original_height = image.size
    
    # 이미지 크기의 80% 계산
    crop_width = int(original_width * 0.8)
    crop_height = int(original_height * 0.8)
    
    # CenterCrop을 사용하여 이미지 중앙에서 80% 크기로 자르기
    return F.center_crop(image, (crop_height, crop_width))


def augment_and_save_images(data_dir, save_dir, num_augmentations):
    # 원본 이미지를 불러옵니다.
    images_path = os.path.join(data_dir, 'train')
    labels_path = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # 증강 변환 설정 (0, 90, 180, 270도 중 하나로 회전)
    augmentation_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply([transforms.RandomChoice([
            transforms.RandomRotation(degrees=[90, 90]),
            transforms.RandomRotation(degrees=[180, 180]),
            transforms.RandomRotation(degrees=[270, 270])
        ])], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)], p=0.2),
        transforms.RandomApply([transforms.Lambda(lambda img: center_crop_80_percent(img))], p=0.3),
        transforms.RandomApply([transforms.RandomInvert()], p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.4)  # 밝기와 대비 조정
    ])
    
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
            transformed_image = augmentation_transforms(image)
            aug_img_save_path = os.path.join(save_dir, f"{i}_aug_{j}.jpg")
            transformed_image.save(aug_img_save_path)
            new_data.append({"image_path": f"{i}_aug_{j}.jpg", "target": label})

    # 새로운 데이터셋을 CSV 파일로 저장
    new_labels_path = os.path.join(save_dir, 'train_augmented.csv')
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(new_labels_path, index=False)
    print(f"New augmented dataset saved at: {new_labels_path}")

# 데이터 증강 적용 및 저장
augment_and_save_images(
    data_dir='/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch',
    save_dir=f'/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch/gpt_aug',
    num_augmentations= 10  # 각 이미지 당 증강 개수
)