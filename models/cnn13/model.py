import torch
from torch import nn

train_mode = 'default'
epochs = 5
lr = 8e-6

class Block3(nn.Module):
    def __init__(self, n_hidden=144):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 3600)
        self.fc3 = nn.Linear(3600, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 3600)
        self.prep1 = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu)
        self.prep2 = nn.Sequential(self.fc3, self.relu, self.fc4, self.relu)
        self.refl_pad = nn.ReflectionPad2d(16)
        self.conv1 = nn.Conv2d(5, 8, 9, padding='valid')
        self.conv2 = nn.Conv2d(8, 8, 9, padding='valid')
        self.conv3 = nn.Conv2d(8, 8, 9, padding='valid')
        self.conv4 = nn.Conv2d(8, 1, 9, padding='valid')
        self.ff = torch.zeros((1, 60, 60))
        self.ff[0, :, 41:51] = 0.5
        self.ff[0, :, 51:] = 1
        self.ff[0, 0, :] = 36.49635

    def forward(self, x, hr):
        # MAE 2.484
        z = torch.empty((x.shape[0], 5, 60, 60), device=x.device)
        z[:, 0, :, :] = x.view(-1, 60, 60)
        z[:, 1, :, :] = self.prep1(x.view(-1, 3600)).view(-1, 60, 60)
        z[:, 2, :, :] = hr.view(-1, 60, 60)
        z[:, 3, :, :] = self.prep2(hr.view(-1, 3600)).view(-1, 60, 60)
        z[:, 4, :, :] = self.ff

        z = self.refl_pad(z)
        z[:, 2, :, :16] = (100-146)/37 - z[:, 2, :, :16]

        r = self.relu(self.conv1(z))
        r = self.relu(self.conv2(r))
        r = self.relu(self.conv3(r))
        r = self.conv4(r)
        return r

class Model(nn.Module):
    #200 epochs lr 8e-6 MAE 2.110
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 128)
        self.fc2 = nn.Linear(128, 3600)
        self.block_1 = Block3(n_hidden=64)
        self.block_2 = Block3(n_hidden=32)
        self.block_3 = Block3(n_hidden=32)


    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = self.relu(self.fc2(h))
        h = h.reshape((-1, 1, 60, 60))
        h = h - self.block_1(x, h)
        h = h - self.block_2(x, h)
        h = h - self.block_3(x, h)
        return h.reshape((-1, 60, 60))

def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    print("Custom training not supported")
    return