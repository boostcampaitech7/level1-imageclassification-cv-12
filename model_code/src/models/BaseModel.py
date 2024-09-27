import torch
import torch.nn as nn
import timm

class EfficientNetV2L(nn.Module):
    def __init__(self, config):
        super(EfficientNetV2L, self).__init__()
        self.config = config
        
        # Load the pre-trained EfficientNetV2-L model from timm
        self.efficientnet = timm.create_model('tf_efficientnetv2_l', pretrained=True)
        
        # Modify the last classifier layer to match 500 classes
        num_classes = 500
        self.efficientnet.reset_classifier(num_classes=num_classes)
    
    def forward(self, x):
        x = self.efficientnet(x)
        return x