import torch.nn as nn
import torch

# architecture du 2ème modèle
class NN2(nn.Module):

    def __init__(self):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(1888, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

