import torch
from torch import nn

train_mode = 'default'
epochs = 100
lr = 2e-4

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 3600)

    def forward(self, x):
        x = x.reshape((-1, 3600))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x.reshape((-1, 60, 60))

def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    print("Custom training not supported")
    return