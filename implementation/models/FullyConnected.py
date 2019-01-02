import torch
import torch.nn as nn
import torch.nn.functional as f


class FullyConnected(nn.Module):
    
    def __init__(self, sequence_length=51):
        super().__init__()
      
        self.fc1 = nn.Linear(sequence_length, 1000)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 800)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(800, 600)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(600, 2)
        
    def forward(self, x):
        x = f.relu(self.dropout1(self.fc1(x)))
        x = f.relu(self.dropout2(self.fc2(x)))
        x = f.relu(self.dropout3(self.fc3(x)))
        x = self.fc4(x)
        return x