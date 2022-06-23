import numpy as np
import torch.nn
import torch.nn.functional


class QNetwork():


    def __init__(self, state_size, action_size, seed):

        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(state_size, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values.""" 
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)