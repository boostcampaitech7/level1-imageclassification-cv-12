import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layer의 입력 크기를 자동으로 계산하도록 수정
        self.fc1_input_features = 128 * (224 // 8) * (224 // 8)
        self.fc1 = nn.Linear(self.fc1_input_features, 512)
        self.fc2 = nn.Linear(512, 500)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)  # 배치 차원을 유지한 채 평탄화
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x