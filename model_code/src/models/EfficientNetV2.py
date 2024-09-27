import torch
import torch.nn as nn
import timm

class EfficientNetV2(nn.Module):
    def __init__(self, config):
        super(EfficientNetV2, self).__init__()
        self.config = config
        
        # 사전 학습된 EfficientNetV2 모델 로드
        self.efficientnet = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        
        # 마지막 분류기 레이어 수정 (500 클래스에 맞게 설정)
        num_classes = 500
        self.efficientnet.reset_classifier(num_classes=num_classes)
    
    def forward(self, x):
        x = self.efficientnet(x)
        return x