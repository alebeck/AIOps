import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import KPIDataset
from models import ConvModel
from util import print_progress_bar


dataset = KPIDataset(
    '../data/test_preprocessed.csv',
    seq_length=1001,
    step_width=1,
    evaluate=True
)

model = ConvModel(1001)
model.load_state_dict(torch.load('./state'))
model = model.cuda()

loader = DataLoader(dataset, 256, False)

iter_per_epoch = len(loader)
result = []

with torch.no_grad():
    for i, x in enumerate(loader):
        x = x.cuda()
        out = model(x).data.cpu().numpy()
        result.extend(list(out.argmax(1)))
        print_progress_bar(i, iter_per_epoch)

df = pd.read_csv('../data/test_preprocessed.csv')
df = df[['KPI ID', 'timestamp', 'value']]

result_df = pd.DataFrame({'predict': result})

submission = df.join(result_df)
submission = submission[['KPI ID', 'timestamp', 'predict', 'value']]

submission.to_csv('./submission.csv')
