import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import pandas as pd

# Hugging Face API 토큰
hf_token = "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"  # 실제 Hugging Face 토큰으로 교체하세요.

# Stable Diffusion 모델 로드
model_id = "stabilityai/stable-diffusion-2-1"

# 이미지 변형 파이프라인 초기화
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=hf_token)

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

# 기본 경로 설정
image_path = "/data/ephemeral/home/level1-imageclassification-cv-12/생성된이미지/KakaoTalk_Photo_2024-09-25-23-15-57.png"
output_dir = "/data/ephemeral/home/level1-imageclassification-cv-12/생성된이미지"
os.makedirs(output_dir, exist_ok=True)

# 텍스트 프롬프트 설정
prompt = "“A highly detailed pencil sketch of the front view of a yellow school bus, showing intricate details of the headlights, grille, and front body. The image should appear as if drawn in grayscale, with soft shading and defined lines, giving it a hand-drawn look like a realistic pencil sketch.”"

# 이미지 처리 및 생성
try:
    # 기존 이미지 로드 및 크기 조정
    init_image = Image.open(image_path).convert("RGB").resize((224, 224))

    # 이미지 생성
    with torch.autocast("cuda"):
        generated_images = pipe(prompt=prompt, image=init_image, strength=0.2, guidance_scale=5.0).images

    # 생성된 이미지 저장 경로 설정
    output_image_path = os.path.join(output_dir, "generated_image.jpg")
    generated_image = generated_images[0].convert("L")  # Grayscale 변환
    generated_image.save(output_image_path)

    # CSV 데이터 추가 (클래스명, 이미지 경로)
    csv_data = [["example_class", output_image_path, "example_class"]]

    # CSV 파일로 저장
    csv_df = pd.DataFrame(csv_data, columns=["class_name", "image_path", "target"])
    csv_df.to_csv(os.path.join(output_dir, "generated_image.csv"), index=False)

    print("이미지 생성 및 CSV 파일 생성 완료!")

except Exception as e:
    print(f"Error processing image {image_path}: {e}")