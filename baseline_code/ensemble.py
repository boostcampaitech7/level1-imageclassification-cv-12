import pandas as pd
from collections import Counter

# CSV 파일 불러오기
df1 = pd.read_csv('./csv/resnet101/kfold_3.csv')
df2 = pd.read_csv('./csv/resnet101/kfold_0.csv')
df3 = pd.read_csv('./csv/resnet101/kfold_1.csv')
df4 = pd.read_csv('./csv/resnet101/kfold_2.csv')
df5 = pd.read_csv('./csv/resnet101/kfold_4.csv')

"""
    ensemble.py

    cutmix_sam_kfold.py 실행을 통해 나온 kfold 의 k개 만큼의 csv를 hard voting을 통해서 앙상블 하는 파일입니다. 

    실행방법 : python ensemble.py 
    
"""
# ID를 기준으로 데이터프레임 병합
merged_df = df1.copy()
merged_df['target_2'] = df2['target']
merged_df['target_3'] = df3['target']
merged_df['target_4'] = df4['target']
merged_df['target_5'] = df5['target']

# 앙상블 적용: 다수결 기반으로 결정, 전부 다르면 첫 번째 CSV 값 사용
def ensemble(row):
    targets = [row['target'], row['target_2'], row['target_3'], row['target_4'], row['target_5']]
    target_count = Counter(targets)
    
    # 가장 흔한 target 값을 찾고, 빈도가 절반 이상이면 그 값 사용
    most_common_target, count = target_count.most_common(1)[0]
    
    if count > 2:  # 절반 이상일 때 선택
        return most_common_target
    else:
        return row['target']  # 첫 번째 CSV 값 사용

# 새로운 앙상블 target 값 생성
merged_df['final_target'] = merged_df.apply(ensemble, axis=1)

# 결과 CSV 저장
merged_df[['ID', 'image_path', 'final_target']].to_csv('./ensemble_output.csv', index=False)
