import os

# 작업할 폴더 경로 지정
target_folder = '/data/ephemeral/home/level1-imageclassification-cv-12/data'

# os.walk()를 사용하여 폴더와 하위 폴더를 재귀적으로 탐색
for root, dirs, files in os.walk(target_folder):
    for filename in files:
        # 파일명이 '._'로 시작하는지 확인
        if filename.startswith('._'):
            file_path = os.path.join(root, filename)
            # 해당 파일 삭제
            os.remove(file_path)
            print(f"삭제됨: {file_path}")

print("모든 '._' 파일이 삭제되었습니다.")
