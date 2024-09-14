import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        self.config = config
        
        # Load the pretrained ResNet-18 model
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer to output 500 classes
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 500)

    def forward(self, x):
        x = self.resnet18(x)
        return x
