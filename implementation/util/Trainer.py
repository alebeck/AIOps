import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss

def fbeta_score(y_true, y_pred, beta, eps=1e-9):
    beta2 = beta**2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum()
    precision = true_positive.div(y_pred.sum().add(eps))
    recall = true_positive.div(y_true.sum().add(eps))

    return torch.mean((precision*recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))


class Trainer:
    
    def __init__(self, model, dataset, batch_size, epochs, optim=Adam, optim_args={}, loss_func=CrossEntropyLoss(),
                 log_nth=10, save_nth=1, save_prefix='', shuffle=True, validation_size=0.2, num_workers=4, collapse_batch=False):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_args = optim_args
        self.loss_func = loss_func
        self.log_nth = log_nth
        self.save_prefix = save_prefix
        self.save_nth = save_nth
        self.num_workers = num_workers
        self.collapse_batch = collapse_batch
        
        # setup data loaders
        idx = list(range(len(self.dataset)))

        if shuffle:
            np.random.shuffle(idx)

        split = int(np.floor(validation_size * len(self.dataset)))
        train_idx, validation_idx = idx[split:], idx[:split]
        
        np.save(f'{self.save_prefix}val_idx.npy', validation_idx)
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(validation_idx)
        
        self.train_loader = DataLoader(self.dataset, self.batch_size, False, train_sampler, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.dataset, self.batch_size, False, val_sampler, num_workers=self.num_workers)
        
    def train(self):
        # set device
        self.model.is_cuda = torch.cuda.is_available()
        if self.model.is_cuda:
            print("Moving model to GPU...")
            self.model.cuda()
            
        # initialize optimizer
        optim = self.optim(self.model.parameters(), **self.optim_args)
        
        # initialize scheduler
        sched = ReduceLROnPlateau(optim, patience=4)
        
        iter_per_epoch = len(self.train_loader)
        
        print('Start training...')
        
        for epoch in range(self.epochs):
            self.model.train()
            
            for i, (x, y) in enumerate(self.train_loader):
                # (x,y) ^= current minibatch
                if self.model.is_cuda:
                    x, y = x.cuda(), y.cuda()

                out = self.model(x)
                if self.collapse_batch:
                    y = y[:, -1]

                optim.zero_grad()
                loss = self.loss_func(out, y)
                loss.backward()
                optim.step()
                
                # logging
                if self.log_nth is not None and i % self.log_nth == 0:
                    print(f"[EPOCH {epoch} BATCH {i}/{iter_per_epoch-1}] TRAIN LOSS: {loss}")
                    
            train_acc = (out.max(1)[1] == y).sum().data.cpu().numpy() / y.shape[0]
            print(f"[FINISHED EPOCH {epoch}] TRAIN ACC/LOSS: {train_acc}/{loss}")
            
            # validation
            self.model.eval()
            loss_sum, acc_sum, f_sum = 0, 0, 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    if self.model.is_cuda:
                        x, y = x.cuda(), y.cuda()
                    
                    out = self.model(x)
                    if self.collapse_batch:
                        y = y[:, -1]
                    loss_sum += self.loss_func(out, y)
                    acc_sum += (out.max(1)[1] == y).sum().data.cpu().numpy() / y.shape[0]
                    f_sum += fbeta_score(y, out.max(1)[1], 1).data.cpu().numpy()

            val_loss = loss_sum / len(self.val_loader)
            val_acc = acc_sum / len(self.val_loader)
            val_f = f_sum / len(self.val_loader)
            print(f"[FINISHED EPOCH {epoch}] VAL f1/acc/loss: {val_f}/{val_acc}/{val_loss}")
            
            if self.save_nth is not None and epoch % self.save_nth == 0:
                print("Saving weights...")
                torch.save(self.model.state_dict(), f'{self.save_prefix}e{epoch}f{round(float(val_f), 6)}.state')
                
            print("")
            
            sched.step(val_loss)
        
        print("Finished training...")