import torch
import torch.nn as nn
import torch.nn.functional as f


class InceptionModel(nn.Module):
    
    def __init__(self, sequence_length=2851):
        super().__init__()
        self.sequence_length = sequence_length
        
        ### Inception 1 ###
        self.pad1_2 = nn.ConstantPad1d(1, 0)
        self.pad1_3 = nn.ConstantPad1d(2, 0)
        
        self.conv1_1 = nn.Conv1d(2, 32, kernel_size=1, stride=1)
        self.conv1_2 = nn.Conv1d(2, 64, kernel_size=3, stride=1)
        self.conv1_3 = nn.Conv1d(2, 64, kernel_size=5, stride=1)
        
        self.pool1 = nn.MaxPool1d(3)
        
        ### Inception 2 ###
        self.pad2_4 = nn.ConstantPad1d(1, 0)
        
        self.conv2_1_1 = nn.Conv1d(160, 32, kernel_size=1, stride=1)
        self.conv2_1_2 = nn.Conv1d(160, 64, kernel_size=1, stride=1)
        self.conv2_1_3 = nn.Conv1d(160, 32, kernel_size=1, stride=1)
        self.pool2_1_4 = nn.MaxPool1d(3, stride=1)
        
        self.pad2_2 = nn.ConstantPad1d(1, 0)
        self.pad2_3 = nn.ConstantPad1d(2, 0)
        
        self.conv2_2_2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.conv2_2_3 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
        self.conv2_2_4 = nn.Conv1d(160, 32, kernel_size=1, stride=1)
        
        self.pool2 = nn.MaxPool1d(3)
        
        ### Inception 3 ###
        self.pad3_4 = nn.ConstantPad1d(1, 0)
        
        self.conv3_1_1 = nn.Conv1d(256, 32, kernel_size=1, stride=1)
        self.conv3_1_2 = nn.Conv1d(256, 64, kernel_size=1, stride=1)
        self.conv3_1_3 = nn.Conv1d(256, 32, kernel_size=1, stride=1)
        self.pool3_1_4 = nn.MaxPool1d(3, stride=1)
        
        self.pad3_2 = nn.ConstantPad1d(1, 0)
        self.pad3_3 = nn.ConstantPad1d(2, 0)
        
        self.conv3_2_2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.conv3_2_3 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
        self.conv3_2_4 = nn.Conv1d(256, 32, kernel_size=1, stride=1)
        
        self.pool3 = nn.MaxPool1d(3)
        
        ### Fully Connected ###
        self.fc1 = nn.Linear(256 * 105 + 26, 1600)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1600, 1000)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1000, 2)
        
    def forward(self, x):
        kpi, x = x[:, :26], x[:, 26:]
        x = x.view((-1, 2, self.sequence_length))

        ### Inception 1 ###
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(self.pad1_2(x))
        x3 = self.conv1_3(self.pad1_3(x))
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = f.relu(self.pool1(x))
        
        ### Inception 2 ###
        x1 = self.conv2_1_1(x)
        x2 = f.relu(self.conv2_1_2(x))
        x3 = f.relu(self.conv2_1_3(x))
        x4 = self.pool2_1_4(self.pad2_4(x))
        
        x2 = self.conv2_2_2(self.pad2_2(x2))
        x3 = self.conv2_2_3(self.pad2_3(x3))
        x4 = self.conv2_2_4(x4)

        x = f.relu(self.pool2(torch.cat([x1, x2, x3, x4], dim=1)))
        
        ### Inception 2 ###
        x1 = self.conv3_1_1(x)
        x2 = f.relu(self.conv3_1_2(x))
        x3 = f.relu(self.conv3_1_3(x))
        x4 = self.pool3_1_4(self.pad3_4(x))
        
        x2 = self.conv3_2_2(self.pad3_2(x2))
        x3 = self.conv3_2_3(self.pad3_3(x3))
        x4 = self.conv3_2_4(x4)

        x = f.relu(self.pool3(torch.cat([x1, x2, x3, x4], dim=1)))
        
        ### Fully Connected ###
        x = x.view((-1, 256 * 105))
        x = torch.cat([kpi, x], dim=1)
        x = self.fc1(x)
        x = f.relu(self.dropout1(x))
        x = self.fc2(x)
        x = f.relu(self.dropout2(x))
        x = self.fc3(x)
        
        return x