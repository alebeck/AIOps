import torch
import torch.nn as nn
import torch.nn.functional as f


class FullyConnected(nn.Module):
    
    def __init__(self, sequence_length=51):
        super().__init__()
      
        self.fc1 = nn.Linear(sequence_length, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 2)
        
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x