import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        
        # Pretrained ResNet18 모델 로드
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # ResNet18의 마지막 fully connected 레이어 수정 (500개의 클래스를 출력하도록)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 500)

    def forward(self, x):
        x = self.resnet(x)
        return x