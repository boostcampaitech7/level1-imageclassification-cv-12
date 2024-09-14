import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class ResNet34(nn.Module):
    def __init__(self, config):
        super(ResNet34, self).__init__()
        self.config = config
        
        # Load the pretrained ResNet-34 model
        self.resnet34 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer to output 500 classes
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, 500)

    def forward(self, x):
        x = self.resnet34(x)
        return x
