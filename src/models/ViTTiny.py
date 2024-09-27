import torch
import torch.nn as nn
import timm

class ViTTiny(nn.Module):
    def __init__(self, config):
        super(ViTTiny, self).__init__()
        self.config = config
        
        # 사전 학습된 ViT Tiny 모델 로드
        self.vit_tiny = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        
        # 마지막 분류기 레이어 수정 (500 클래스에 맞게 설정)
        num_classes = 500
        self.vit_tiny.reset_classifier(num_classes=num_classes)
    
    def forward(self, x):
        x = self.vit_tiny(x)
        return x