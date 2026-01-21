import torch
from torch import nn

train_mode = 'default2'
#training_settings = {"epochs": 200,
#                     "lr": 8e-6,
#                     "postfix": ""}
#training_settings = {"epochs": 200,
#                     "lr": 1e-6,
#                     "postfix": "lr1e-6"}
#training_settings = {"epochs": 200,
#                     "lr": 6e-5,
#                     "postfix": "_lr6e-5"}
training_settings = {"epochs": 200,
                     "lr": 5e-4,
                     "postfix": "_lr5e-4"}


class Block2(nn.Module):
    def __init__(self, n_hidden=144):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 3600)
        self.fc3 = nn.Linear(3600, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 3600)
        self.prep1 = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu)
        self.prep2 = nn.Sequential(self.fc3, self.relu, self.fc4, self.relu)
        self.conv1 = nn.Conv2d(4, 8, 9, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(8, 1, 9, padding='same', padding_mode='zeros')

    def forward(self, x, hr):
        z = torch.empty((x.shape[0], 4, 60, 60), device=x.device)
        z[:, 0, :, :] = x.view(-1, 60, 60)
        z[:, 1, :, :] = self.prep1(x.view(-1, 3600)).view(-1, 60, 60)
        z[:, 2, :, :] = hr.view(-1, 60, 60)
        z[:, 3, :, :] = self.prep2(hr.view(-1, 3600)).view(-1, 60, 60)

        r = self.relu(self.conv1(z))
        r = self.relu(self.conv2(r))
        r = self.relu(self.conv3(r))
        r = self.conv4(r)
        return r

class Model(nn.Module):
    # cnn12: this model uses four blocks and a double fully connected layes. The idea is that the first 2 fully
    # connected layers compute a estimate of the h we want to predict, and the blocks iteratively compute a refinement to
    # this estimate. Each block has four convolutional layers. The input to these convolutional layers consists of four
    # channels, one for the raw input (the conductivity), one for the most recent estimate of h, and the other two
    # channels use the same date but first apply a double fully connected layer to help generate the global structure
    # of the h-field. We used double fully connected layers instead of single ones because this reduces the number of
    # parameters because we ues a small number of hidden layers
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 32)
        self.fc2 = nn.Linear(32, 3600)
        self.block_1 = Block2(n_hidden=32)
        self.block_2 = Block2(n_hidden=64)
        self.block_3 = Block2(n_hidden=64)
        self.block_4 = Block2(n_hidden=128)


    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = self.relu(self.fc2(h))
        h = h.reshape((-1, 1, 60, 60))
        h = h - self.block_1(x, h)
        h = h - self.block_2(x, h)
        h = h - self.block_3(x, h)
        h = h - self.block_4(x, h)
        return h.reshape((-1, 60, 60))

def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    print("Custom training not supported")
    return