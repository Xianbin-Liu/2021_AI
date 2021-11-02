import numpy as np
from numpy import TooHardError
import pandas as pd 
import torch
from torch._C import LiteScriptModule, import_ir_module
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
    

class HumanRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1) -> None:
        super(HumanRegression, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        output = self.model(x)
        return output
    

    def train(self, X, Y, epoch=10, lr=1e-2, loss_function=nn.BCELoss(), batch_size=100, shuffle=True):
        Xtorch = torch.FloatTensor(X)
        Ytorch = torch.FloatTensor(Y).reshape(-1,1)
        # X = torch.FloatTensor(X)
        # Y = torch.FloatTensor(Y)
        def _custom_collate_fn(batch):
            x, y = zip(*batch)
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y).reshape((-1, 1))
            return x, y
        # Set the number of epoch, which determines the number of training iterations
        n_epoch = epoch
        adam = optim.Adam(self.parameters(), lr=lr)
        loader = DataLoader(list(zip(X,Y)), batch_size=batch_size, shuffle=shuffle, collate_fn=_custom_collate_fn)
        for epoch in range(n_epoch):
            for batch_x, batch_y in loader:
                # Set the gradients to 0
                adam.zero_grad()
                # Get the model predictions
                y_pred = self.forward(batch_x)
                # Get the loss
                loss = loss_function(y_pred, batch_y)
                # Print stats
                # Compute the gradients
                loss.backward()

                # Take a step to optimize the weights
                adam.step()
            y_pred = self.forward(Xtorch)
            ac = (np.array((y_pred > 0.5) == Ytorch)).mean()
            print(f"Epoch {epoch}: traing loss: {loss}")
            print(f"Epoch {epoch}: accuarracy: {ac}")



if __name__ == '__main__':
    dataset = pd.read_csv("check/train.csv", header=None).values
    dataset[:,-1] = dataset[:,-1].astype('int')
    valset = dataset[-1000:, :]
    valabel = valset[:,-1].reshape((-1,1))
    valset = torch.FloatTensor(dataset[-1000:,:-1])
    model = HumanRegression(40, 10, 1)
    model.train(dataset[:,:-1], dataset[:,-1])
    yp = model.forward(valset)
    yp = np.array(yp > 0.5)
    ac = (yp == valabel).mean()
    print(ac)