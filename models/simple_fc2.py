import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import no_grad
import numpy as np

class simple_fc2(nn.Module):

    def __init__(self, indims, outdims, lr, wd, tr_epochs):
        super().__init__()
        self.indims = indims
        self.middledims = int(indims * 0.25)
        self.outdims = outdims
        self.lr = lr
        self.wd = wd
        self.tr_epochs = tr_epochs
        self.criterion = nn.MSELoss
        self.optimizer = optim.Adam
        self.fc1 = nn.Linear(self.indims, self.middledims)
        self.fc2 = nn.Linear(self.middledims, self.middledims)
        self.fc3 = nn.Linear(self.middledims, self.outdims)

    def learn(self, x, y):
        criterion = self.criterion()
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.wd)
        for i in range(self.tr_epochs):
            pred = self.forward(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i, "\n=============")
        return loss.item()

    def forward(self, x):
        out1 = F.tanh(self.fc1(x))      # in-->middle, tanh
        out2 = F.relu(self.fc2(out1))   # middle, Relu
        out3 = self.fc3(out2)                # middle --> out
        return out3

    def predict(self, x):
        out = None
        with no_grad():
            out = self.forward(x).numpy()
        return out
