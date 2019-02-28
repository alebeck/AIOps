import torch
import torch.nn as nn
import torch.nn.functional as f

class ConvModel(nn.Module):

	def __init__(self, sequence_length=1001):
		super().__init__()
		self.sequence_length = sequence_length
	  
		self.pad1 = nn.ConstantPad1d(1, 0)
		self.conv1 = nn.Conv1d(2, 32, kernel_size=3, stride=1)
		self.pool1 = nn.MaxPool1d(2)
		
		self.pad2 = nn.ConstantPad1d(1, 0)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
		self.pool2 = nn.MaxPool1d(2)
		
		self.pad3 = nn.ConstantPad1d(1, 0)
		self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1)
		self.pool3 = nn.MaxPool1d(2)
		
		# one hot vector for ID, 788 statistical data per ID
		self.fc1 = nn.Linear(128 * 123 + 26 + 788 * 2, 1200)
		self.fc2 = nn.Linear(1200, 800)
		self.fc3 = nn.Linear(800, 2)
		
	def forward(self, x):
		kpi, x = x[:, :(26 + 788 * 2)], x[:, (26 + 788 * 2):]
		x = x.view((-1, 2, self.sequence_length))

		x = self.pad1(x)
		x = f.relu(self.conv1(x))
		x = self.pool1(x)
		
		x = self.pad2(x)
		x = f.relu(self.conv2(x))
		x = self.pool2(x)
		
		x = self.pad3(x)
		x = f.relu(self.conv3(x))
		x = self.pool3(x)
		
		x = x.view((-1, 128 * 123))
		x = torch.cat([kpi, x], dim=1)
		x = f.relu(self.fc1(x))
		x = f.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x