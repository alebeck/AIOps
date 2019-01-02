import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append('./implementation/')
from implementation.util.KPIStatsDataset import KPIStatsDataset
from implementation.models.ConvModel import ConvModel


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()

dataset = KPIStatsDataset(
    '../data/test_preprocessed.csv', 
    './stats/KPI_ID_imp_series_features.csv', 
    './stats/KPI_ID_imp_diff_features.csv',
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