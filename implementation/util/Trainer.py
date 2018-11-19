import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss


class Trainer:
    
    def __init__(self, model, dataset, batch_size, epochs, optim=Adam, optim_args={}, loss_func=CrossEntropyLoss(),
                 log_nth=10, shuffle=True, validation_size=0.2, num_workers=4):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_args = optim_args
        self.loss_func = loss_func
        self.log_nth = log_nth
        self.num_workers = num_workers
        
        # setup data loaders
        idx = list(range(len(self.dataset)))

        if shuffle:
            np.random.shuffle(idx)

        split = int(np.floor(validation_size * len(self.dataset)))
        train_idx, validation_idx = idx[split:], idx[:split]
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(validation_idx)
        
        self.train_loader = DataLoader(self.dataset, self.batch_size, False, train_sampler, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.dataset, self.batch_size, False, val_sampler, num_workers=self.num_workers)
        
    def train(self):
        # initialize optimizer
        optim = self.optim(self.model.parameters(), **self.optim_args)

        # initialize scheduler
        sched = ReduceLROnPlateau(optim, patience=5)

        # set device
        self.model.is_cuda = torch.cuda.is_available()
        if self.model.is_cuda:
            print("Moving model to GPU...")
            self.model.cuda()
        
        iter_per_epoch = len(self.train_loader)
        
        print('Start training...')
        
        for epoch in range(self.epochs):
            self.model.train()
            
            for i, (x, y) in enumerate(self.train_loader):
                # (x,y) ^= current minibatch
                if self.model.is_cuda:
                    x, y = x.cuda(), y.cuda()

                out = self.model(x)
                optim.zero_grad()
                loss = self.loss_func(out, y)
                loss.backward()
                optim.step()
                
                # logging
                if self.log_nth is not None and i % self.log_nth == 0:
                    print(f"[EPOCH {epoch} BATCH {i}/{iter_per_epoch-1}] TRAIN LOSS: {loss}")
                    
            train_acc = (out.max(1)[1] == y).sum().data.numpy() / y.shape[0]
            print(f"[FINISHED EPOCH {epoch}] TRAIN ACC/LOSS: {train_acc}/{loss}")
            
            # validation
            self.model.eval()
            loss_sum, acc_sum = 0, 0
            for x, y in self.val_loader:
                out = self.model(x)
                loss_sum += self.loss_func(out, y)
                acc_sum += (out.max(1)[1] == y).sum().data.numpy() / y.shape[0]

            val_loss = loss_sum / len(self.val_loader)
            val_acc = acc_sum / len(self.val_loader)
            print(f"[FINISHED EPOCH {epoch}] VAL acc/loss: {val_acc}/{val_loss}\n")
            
            sched.step(val_loss)
        
        print("Finished training...")