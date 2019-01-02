import sys
from torch.nn import CrossEntropyLoss

sys.path.append('./implementation/')
from implementation.util.Trainer import Trainer
from implementation.util.KPIStatsDataset import KPIStatsDataset
from implementation.models.ConvModel import ConvModel


dataset = KPIStatsDataset(
    '../data/train_preprocessed.csv', 
    './stats/KPI_ID_imp_series_features.csv', 
    './stats/KPI_ID_imp_diff_features.csv',
    seq_length=1001,
    step_width=1
)

model = ConvModel(1001)

args = {
    "lr": 0.5e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0
}

trainer = Trainer(
    model,
    dataset,
    batch_size=512,
    epochs=100,
    log_nth=800,
    validation_size=0.19,
    optim_args=args,
    loss_func=CrossEntropyLoss()
)

trainer.train()