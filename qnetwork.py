import numpy as np
import torch.nn
import torch.nn.functional
import torch.optim
import torch


class QNetwork(torch.nn.Module):


    def __init__(self,lr, state_size, action_size, seed):

        super(QNetwork, self).__init__()
        
       # self.seed = torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(*state_size, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, action_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        
    def forward(self, state):
        """Build a network that maps state -> action values.""" 
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

