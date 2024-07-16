import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as that
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 n_units=100,
                 epochs=1000,
                 loss=nn.MSELoss(),
                 lr=1e-3,
                 loss2=None,
                 loss2_weight=0.1) -> None:
        
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

        self.auto_set_device()

    def auto_set_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        #Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            
    def get_device(self):
        return self.device

    def set_device(self, input):
        __allowed_keywords = ["cpu", "cuda"]
        
        if input in __allowed_keywords:
            self.device = input
        else:
            raise AttributeError(f"Cannot set unknown attribute '{input}'")
    
    ## Seed for cuda and cpu
    def fix_seed(self, seed_no=42):
        self.seed = seed_no
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
        else:
            torch.manual_seed(self.seed)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def np_to_th(self,x):
        n_samples = len(x)
        return torch.from_numpy(x).to(torch.float).to(self.device).reshape(n_samples, -1)

    def fit(self, X, y):
        Xt = self.np_to_th(X)
        yt = self.np_to_th(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight + self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def fit_tqdm(self, X, y):
        Xt = self.np_to_th(X)
        yt = self.np_to_th(y)
    
        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        
        # Wrap the loop with tqdm
        for ep in tqdm(range(self.epochs), desc="Training Progress"):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight + self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())       
        return losses

    
    def predict(self, X):
        self.eval()
        out = self.forward(self.np_to_th(X))
        return out.detach().cpu().numpy()
    

class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))

def l2_reg(model: torch.nn.Module):
    return torch.sum(sum([p.pow(2.) for p in model.parameters()]))