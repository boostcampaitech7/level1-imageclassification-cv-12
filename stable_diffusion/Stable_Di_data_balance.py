import pandas as pd


'''

    생성된 이미지의 csv와 실제 csv를 하나로 합쳐서 최종적으로 Train 가능하도록 csv를 생성하게 됩니다.

'''

# 각 CSV 파일 경로
generated_images_csv_path = "/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch/생성된이미지/generated_images.csv"
train_csv_path = "/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch/train.csv"

# CSV 파일 불러오기
generated_images_df = pd.read_csv(generated_images_csv_path)
train_df = pd.read_csv(train_csv_path)

# 두 CSV 파일을 하나로 합치기
combined_df = pd.concat([generated_images_df, train_df], ignore_index=True)

# 합쳐진 CSV 파일을 저장할 경로
combined_csv_path = "/data/ephemeral/home/level1-imageclassification-cv-12/data/sketch/combined_images.csv"

# 합쳐진 CSV 파일 저장
combined_df.to_csv(combined_csv_path, index=False)

print(f"CSV 파일이 성공적으로 합쳐졌습니다. 저장 경로: {combined_csv_path}")