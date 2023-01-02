# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class FireDetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # x = x.view(-1, 3, 8, 8)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class mod(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.model = torch.nn.Sequential(
			#Input = 3 x 32 x 32, Output = 32 x 32 x 32
			torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),
			torch.nn.ReLU(),
			#Input = 32 x 32 x 32, Output = 32 x 16 x 16
			torch.nn.MaxPool2d(kernel_size=2),

			#Input = 32 x 16 x 16, Output = 64 x 16 x 16
			torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
			torch.nn.ReLU(),
			#Input = 64 x 16 x 16, Output = 64 x 8 x 8
			torch.nn.MaxPool2d(kernel_size=2),
			
			#Input = 64 x 8 x 8, Output = 64 x 8 x 8
			torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
			torch.nn.ReLU(),
			#Input = 64 x 8 x 8, Output = 64 x 4 x 4
			torch.nn.MaxPool2d(kernel_size=2),

			torch.nn.Flatten(),
			torch.nn.Linear(64*4*4, 512),
			torch.nn.ReLU(),
			torch.nn.Linear(512, 10)
		)

	def forward(self, x):
		return self.model(x)



model = FireDetectionNet()
model = mod()

model(torch.rand(size=(32,3,224,224)))