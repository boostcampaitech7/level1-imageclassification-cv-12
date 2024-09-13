import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

class EfficientNetB1(nn.Module):
    def __init__(self, config):
        super(EfficientNetB1, self).__init__()
        self.config = config
        
        # Pretrained EfficientNet B1 모델 로드
        self.efficientnet = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        
        # EfficientNet B1의 마지막 fully connected 레이어 수정 (500개의 클래스를 출력하도록)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, 500)

    def forward(self, x):
        x = self.efficientnet(x)
        return x