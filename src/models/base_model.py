import torch
import torch.nn.functional as F
from torch import nn

# 실제 학습을 시킬 베이스 모델을 만들어둔다.
class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

        # 224 * 224 이미지에 맞춘 첫 번째 Linear layer
        self.layer_1 = nn.Linear(224 * 224 * 3, 1024)  # 입력 채널 3 (RGB 이미지)
        self.layer_2 = nn.Linear(1024, 512)  # 중간 레이어
        self.layer_3 = nn.Linear(512, 500)   # 출력 클래스 수: 500

    def forward(self, x):
        # 입력을 flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x