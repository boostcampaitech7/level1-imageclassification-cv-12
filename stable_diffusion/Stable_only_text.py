import torch
from diffusers import StableDiffusionPipeline
import os
import pandas as pd


'''
    해당 파일은 텍스트를 활용하여 원하는 이미지를 생성할 서 있는 stable diffusion 파일입니다.

    원하는 prompt를 입력하여 새로운 이미지를 활용하게 됩니다.
'''

# Hugging Face API 토큰
hf_token = ""  # Replace with your actual Hugging Face API token.

# Stable Diffusion 모델 로드
model_id = "stabilityai/stable-diffusion-2-1"

# 이미지 변형 파이프라인 초기화
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=hf_token)

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# 기본 경로 설정
output_dir = "/data/ephemeral/home/level1-imageclassification-cv-12/생성된이미지"
os.makedirs(output_dir, exist_ok=True)

# 프롬포트 작성
prompt = ''


# 이미지 생성 및 저장 (10개 생성)
csv_data = []
try:
    for i in range(10):
        with torch.autocast("cuda"):
            generated_images = pipe(prompt=prompt, guidance_scale=7.5).images  # 이미지 생성

        # 생성된 이미지 저장 경로 설정
        output_image_path = os.path.join(output_dir, f"generated_{i+1}.jpg")
        generated_image = generated_images[0]
        generated_image.save(output_image_path)

        # CSV 데이터 추가 (이미지 경로)
        csv_data.append(["", output_image_path, ""])

        print(f"Generated and saved image {i+1}/10")

    # CSV 파일로 저장
    csv_df = pd.DataFrame(csv_data, columns=["class_name", "image_path", "target"])
    csv_df.to_csv(os.path.join(output_dir, "generated_images.csv"), index=False)

    print("모든 이미지 생성 및 CSV 파일 생성 완료!")

except Exception as e:
    print(f"Error processing image: {e}")