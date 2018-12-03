import torch
import torch.nn as nn
import torch.nn.functional as f


class LSTM_LA(nn.Module):
    
    def __init__(self, hidden_size=400):
        super().__init__()
        self.hidden_size = hidden_size
      
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        # TODO: LA through fully connected layer
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.lstm(x)
        x = x[0].contiguous().view((-1, self.hidden_size))
        x = self.fc(x)
        return x