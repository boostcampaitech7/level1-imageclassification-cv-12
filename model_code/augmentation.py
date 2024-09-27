import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import torchvision.transforms.functional as F


def augment_and_save_images(data_dir, save_dir):
    # 원본 이미지를 불러옵니다.
    images_path = os.path.join(data_dir, 'train')
    labels_path = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
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
        
        # 상하 flip 이미지 생성 및 저장
        v_flipped_image = F.vflip(image)
        v_flipped_img_save_path = os.path.join(save_dir, f"{i}_v_flipped.jpg")
        v_flipped_image.save(v_flipped_img_save_path)
        new_data.append({"image_path": f"{i}_v_flipped.jpg", "target": label})
        
        # 좌우 flip 이미지 생성 및 저장
        h_flipped_image = F.hflip(image)
        h_flipped_img_save_path = os.path.join(save_dir, f"{i}_h_flipped.jpg")
        h_flipped_image.save(h_flipped_img_save_path)
        new_data.append({"image_path": f"{i}_h_flipped.jpg", "target": label})

        # 상하좌우 flip 이미지 생성 및 저장
        vh_flipped_image = F.hflip(v_flipped_image)
        vh_flipped_img_save_path = os.path.join(save_dir, f"{i}_vh_flipped.jpg")
        vh_flipped_image.save(vh_flipped_img_save_path)
        new_data.append({"image_path": f"{i}_vh_flipped.jpg", "target": label})

        # 180도 회전 이미지 생성 및 저장
        rotated_180_image = F.rotate(image, angle=180)
        rotated_180_img_save_path = os.path.join(save_dir, f"{i}_rotated_180.jpg")
        rotated_180_image.save(rotated_180_img_save_path)
        new_data.append({"image_path": f"{i}_rotated_180.jpg", "target": label})

        # 색상 반전 이미지 생성 및 저장
        inverted_image = F.invert(image)
        inverted_img_save_path = os.path.join(save_dir, f"{i}_inverted.jpg")
        inverted_image.save(inverted_img_save_path)
        new_data.append({"image_path": f"{i}_inverted.jpg", "target": label})

        # 80% 크롭 이미지 생성 및 저장
        crop_transform = transforms.RandomResizedCrop(size=image.size, scale=(0.8, 0.8))
        cropped_image = crop_transform(image)
        cropped_img_save_path = os.path.join(save_dir, f"{i}_cropped.jpg")
        cropped_image.save(cropped_img_save_path)
        new_data.append({"image_path": f"{i}_cropped.jpg", "target": label})

        # 0 ~ 10도 회전 이미지 생성 및 저장
        rotated_image = F.rotate(image, angle=10)
        rotated_img_save_path = os.path.join(save_dir, f"{i}_rotated.jpg")
        rotated_image.save(rotated_img_save_path)
        new_data.append({"image_path": f"{i}_rotated.jpg", "target": label})

    # 새로운 데이터셋을 CSV 파일로 저장
    new_labels_path = os.path.join(save_dir, 'train_augmented.csv')
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(new_labels_path, index=False)
    print(f"New augmented dataset with horizontal flip, vertical flip, VH flip, color inversion, and cropping saved at: {new_labels_path}")

# 데이터 증강 적용 및 저장 (원본 + 상하 flip, 좌우 flip, 상하좌우 flip, 색상 반전, 80% 크롭, 0 ~ 10도 회전)
augment_and_save_images(
    data_dir='/data/ephemeral/home/JSM_Secret_Code/data/sketch/data',
    save_dir=f'/data/ephemeral/home/JSM_Secret_Code/data/sketch/data/simple'
)


