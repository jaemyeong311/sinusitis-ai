import torch.nn as nn
CLASS_ORDER=['Normal','Left','Right','Both']
class SinusitisLightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.BatchNorm2d(32),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.BatchNorm2d(64),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.c=nn.Sequential(nn.Flatten(),nn.Linear(128,128),nn.ReLU(),nn.Dropout(.5),nn.Linear(128,4))
    def forward(self,x):
        return self.c(self.f(x))
