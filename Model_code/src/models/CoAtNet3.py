import torch
import torch.nn as nn
import timm

class CoAtNet3(nn.Module):
    def __init__(self, config):
        super(CoAtNet3, self).__init__()
        self.config = config
        
        self.coatnet = timm.create_model('coatnet_3_rw_224', pretrained=True)
        
       
        num_classes = 500  
        self.coatnet.reset_classifier(num_classes=num_classes)
    
    def forward(self, x):
        x = self.coatnet(x)
        return x


