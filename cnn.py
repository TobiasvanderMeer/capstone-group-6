import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils import load_files
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, padding='same')  # -> 100x100
        self.conv2 = nn.Conv2d(5, 3, 7, stride=3, padding='valid')  # -> 32x32
        self.conv3 = nn.Conv2d(3, 2, 5, stride=3, padding='valid')  # -> 10x10
        self.fc1 = nn.Linear(200, 100)  # -> 10x10
        self.conv4 = nn.Conv2d(3, 3, 5, padding='same')  # -> 32x32
        self.conv5 = nn.Conv2d(6, 3, 7, padding='same')  # -> 100x100
        self.conv6 = nn.Conv2d(8, 4, 7, padding='same')  # -> 100x100
        self.conv7 = nn.Conv2d(4, 1, 5, padding='same')  # -> 100x100
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.fc1(x3.flatten(start_dim=1))).reshape((-1, 1, 10, 10))
        x5 = self.relu(self.conv4(torch.cat((x4, 0*x3), dim=1)))
        x5 = nn.functional.interpolate(x5, size=(32, 32))
        x6 = self.relu(self.conv5(torch.cat((x5, 0*x2), dim=1)))
        x6 = nn.functional.interpolate(x6, size=(100, 100))
        x7 = self.relu(self.conv6(torch.cat((x6, 0.5*x1), dim=1)))
        y = self.conv7(x7)[:, 0, :, :]
        return y

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 144)
        self.fc2 = nn.Linear(144, 144)
        self.fc3 = nn.Linear(144, 3600)

    def forward(self, x):
        x = x.reshape((-1, 3600))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.reshape((-1, 60, 60))


class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.fc2 = nn.Linear(3600, 3600)
        self.fc3 = nn.Linear(3600, 3600)
        self.fc4 = nn.Linear(3600, 3600)
        self.fc5 = nn.Linear(3600, 3600)

    def forward(self, x):
        x = x.reshape((-1, 3600))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x.reshape((-1, 60, 60))


class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.fc_p = nn.Linear(3600, 3600)
        self.conv_p1 = nn.Conv2d(2, 5, 9, padding='same')
        self.conv_p2 = nn.Conv2d(5, 1, 9, padding='same')

    def powerlayer(self, x, h):
        #h = self.relu(self.fc_p(h))
        h2 = torch.cat((x, h), dim=1)
        h2 = self.relu(self.conv_p1(h2))
        h2 = self.conv_p2(h2)
        return h2


class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.conv_p1 = nn.Conv2d(2, 8, 9, padding='same')
        self.conv_p2 = nn.Conv2d(8, 8, 9, padding='same')
        self.conv_p3 = nn.Conv2d(8, 8, 9, padding='same')
        self.conv_p4 = nn.Conv2d(8, 1, 9, padding='same')

    def powerlayer(self, x, h):
        # h = self.relu(self.fc_p(h))
        h2 = torch.cat((x, h), dim=1)
        h2 = self.relu(self.conv_p1(h2))
        h2 = self.relu(self.conv_p2(h2))
        h2 = self.relu(self.conv_p3(h2))
        h2 = self.conv_p4(h2)
        return h2

    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = h.reshape((-1, 1, 60, 60))
        for _ in range(5):
            h = self.powerlayer(x, h)
            #h[:, :, :, 0] = 100
            #h[:, :, :, -2:] = torch.mean(h[:, :, :, -2:], dim=3, keepdim=True)
            #h[:, :, -2:, :] = torch.mean(h[:, :, -2:, :], dim=2, keepdim=True)
            #print(torch.exp(x[:, :, 0, :]+4).shape)
            #print((torch.tensor([-1, 1])[None, None, :, None]  * -500).shape)
            #h[:, :, :2, :] = (torch.mean(h[:, :, :2, :], dim=2, keepdim=True) +
            #                  torch.tensor([-1, 1])[None, None, :, None] * torch.exp(x[:, :, 0, :].reshape((-1, 1, 1, 60))+4) * -500*0.1)
        return h.reshape((-1, 60, 60))

class Model6(nn.Module):
    ff = torch.zeros((1, 1, 60, 60))
    ff[0, 0, :, 41:51] = 0.5
    ff[0, 0, :, 51:] = 1

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.conv_p1 = nn.Conv2d(3, 8, 9, padding='valid')
        self.conv_p2 = nn.Conv2d(8, 8, 9, padding='valid')
        self.conv_p3 = nn.Conv2d(8, 8, 9, padding='valid')
        self.conv_p4 = nn.Conv2d(8, 1, 9, padding='valid')

    def powerlayer(self, x, h):
        # h = self.relu(self.fc_p(h))
        h2 = torch.zeros((x.shape[0]) ,3, 92, 92)
        h2[:, :, 16:76, 16:76] = torch.cat((x, h, self.ff.expand(x.shape)), dim=1)
        h2[:, :, 16:76, 76:] = h2[:, :, 16:76, 60:76].flip(3)
        h2[:, :, 76:, 16:] = h2[:, :, 60:76, 16:].flip(2)

        h2[:, :, :16, 16:] = 0

        h2[:, 0, :, :16] = h2[:, 0, :, 16:32].flip(2)
        h2[:, 2, :, :16] = h2[:, 2, :, 16:32].flip(2)
        h2[:, 1, :, :16] = -1.2432432432432432-h2[:, 1, :, 16:32].flip(2)

        h2 = self.relu(self.conv_p1(h2))
        h2 = self.relu(self.conv_p2(h2))
        h2 = self.relu(self.conv_p3(h2))
        h2 = self.conv_p4(h2)
        return h2

    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = h.reshape((-1, 1, 60, 60))
        for _ in range(5):
            h = self.powerlayer(x, h)
            #h[:, :, :, 0] = 100
            #h[:, :, :, -2:] = torch.mean(h[:, :, :, -2:], dim=3, keepdim=True)
            #h[:, :, -2:, :] = torch.mean(h[:, :, -2:, :], dim=2, keepdim=True)
            #print(torch.exp(x[:, :, 0, :]+4).shape)
            #print((torch.tensor([-1, 1])[None, None, :, None]  * -500).shape)
            #h[:, :, :2, :] = (torch.mean(h[:, :, :2, :], dim=2, keepdim=True) +
            #                  torch.tensor([-1, 1])[None, None, :, None] * torch.exp(x[:, :, 0, :].reshape((-1, 1, 1, 60))+4) * -500*0.1)
        return h.reshape((-1, 60, 60))


class Model7(nn.Module):
    #best of the powerlayer models
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.fc2 = nn.Linear(3600, 3600)
        self.conv_p1 = nn.Conv2d(2, 8, 9, padding='same', padding_mode='zeros')
        self.conv_p2 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv_p3 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv_p4 = nn.Conv2d(8, 1, 9, padding='same', padding_mode='zeros')

    def powerlayer(self, x, h):
        # h = self.relu(self.fc_p(h))
        h2 = torch.cat((x, h), dim=1)
        h2 = self.relu(self.conv_p1(h2))
        h2 = self.relu(self.conv_p2(h2))
        h2 = self.relu(self.conv_p3(h2))
        h2 = self.conv_p4(h2)  # test relu here
        return h2

    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = h.reshape((-1, 1, 60, 60))
        for _ in range(5):
            h = self.relu(self.fc2(h.view((-1, 3600)))).view((-1, 1, 60, 60))
            h = self.powerlayer(x, h)
        return h.reshape((-1, 60, 60))


class Model8(nn.Module):
    #trains less good than model7, but might get better performance after more epochs
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.fc2 = nn.Linear(3600, 3600)
        self.conv_p1 = nn.Conv2d(2, 8, 9, padding='same', padding_mode='zeros')
        self.conv_p2 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv_p3 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv_p4 = nn.Conv2d(8, 1, 9, padding='same', padding_mode='zeros')
        self.conv_f1 = nn.Conv2d(1, 2, 9, padding='same', padding_mode='replicate')
        self.conv_f2 = nn.Conv2d(2, 1, 7, padding='same', padding_mode='replicate')

    def powerlayer(self, x, h):
        # h = self.relu(self.fc_p(h))
        h2 = torch.cat((x, h), dim=1)
        h2 = self.relu(self.conv_p1(h2))
        h2 = self.relu(self.conv_p2(h2))
        h2 = self.relu(self.conv_p3(h2))
        h2 = self.conv_p4(h2)
        return h2

    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = h.reshape((-1, 1, 60, 60))
        for _ in range(5):
            h = self.relu(self.fc2(h.view((-1, 3600)))).view((-1, 1, 60, 60))
            h = self.powerlayer(x, h)
        h = self.relu(self.conv_f1(h))
        h = self.conv_f2(h)
        return h.reshape((-1, 60, 60))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.fc2 = nn.Linear(3600, 3600)
        self.conv1 = nn.Conv2d(4, 8, 9, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(8, 1, 9, padding='same', padding_mode='zeros')

    def forward(self, x, hr):
        z = torch.empty((x.shape[0], 4, 60, 60))
        z[:, 0, :, :] = x.view(-1, 60, 60)
        z[:, 1, :, :] = self.relu(self.fc1(x.view(-1, 3600))).view(-1, 60, 60)
        z[:, 2, :, :] = hr.view(-1, 60, 60)
        z[:, 3, :, :] = self.relu(self.fc2(hr.view(-1, 3600)).view(-1, 60, 60))

        r = self.relu(self.conv1(z))
        r = self.relu(self.conv2(r))
        r = self.relu(self.conv3(r))
        r = self.conv4(r)
        return r

class Model9(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.block_1 = Block()
        self.block_2 = Block()
        self.block_3 = Block()


    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = h.reshape((-1, 1, 60, 60))
        h = self.relu(self.block_1(x, h))
        h = self.relu(self.block_2(x, h))
        h = self.block_3(x, h)
        return h.reshape((-1, 60, 60))

class Model10(nn.Module):
    #trains OK (12 epochs 7600 samples) MAE 5.33788 minor sign of overfitting
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.block_1 = Block()
        self.block_2 = Block()
        self.block_3 = Block()


    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = h.reshape((-1, 1, 60, 60))
        h = h - self.block_1(x, h)
        h = h - self.block_2(x, h)
        h = h - self.block_3(x, h)
        return h.reshape((-1, 60, 60))

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

class Model11(nn.Module):
    #best so far converges nicely likely better performance with lower learning rate and more epochs no overfitting
    #(12 epochs 7600 samples) lr 3e-5 MAE 3.893

    #(100 epochs 7600 samples) lr 1e-5 MAE 2.256    lr 3e-6 is worse
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 144)
        self.fc2 = nn.Linear(144, 3600)
        self.block_1 = Block2(n_hidden=144)
        self.block_2 = Block2(n_hidden=144)
        self.block_3 = Block2(n_hidden=225)


    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = self.relu(self.fc2(h))
        h = h.reshape((-1, 1, 60, 60))
        h = h - self.block_1(x, h)
        h = h - self.block_2(x, h)
        h = h - self.block_3(x, h)
        return h.reshape((-1, 60, 60))


class Model12(nn.Module):
    #MAE 2.059 (150 epochs 7600 samples) could improve with more training
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

class Block2b(nn.Module):
    def __init__(self, n_hidden=144):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 3600)
        self.fc3 = nn.Linear(3600, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 3600)
        self.prep1 = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu)
        self.prep2 = nn.Sequential(self.fc3, self.relu, self.fc4, self.relu)
        self.conv1 = nn.Conv2d(4, 16, 9, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(16, 64, 9, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv2d(64, 64, 9, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(64, 8, 9, padding='same', padding_mode='zeros')
        self.conv5 = nn.Conv2d(8, 1, 9, padding='same', padding_mode='zeros')

    def forward(self, x, hr):
        z = torch.empty((x.shape[0], 4, 60, 60), device=x.device)
        z[:, 0, :, :] = x.view(-1, 60, 60)
        z[:, 1, :, :] = self.prep1(x.view(-1, 3600)).view(-1, 60, 60)
        z[:, 2, :, :] = hr.view(-1, 60, 60)
        z[:, 3, :, :] = self.prep2(hr.view(-1, 3600)).view(-1, 60, 60)

        r = self.relu(self.conv1(z))
        r = self.relu(self.conv2(r))
        r = self.relu(self.conv3(r))
        r = self.relu(self.conv4(r))
        r = self.conv5(r)
        return r


class Model12b(nn.Module):
    #first 30 epochs at lr 1e-5, then 30 at 3e-6
    #MAE 2.1407
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 128)
        self.fc2 = nn.Linear(128, 3600)
        self.block_1 = Block2b(n_hidden=128)
        self.block_2 = Block2b(n_hidden=64)
        self.block_3 = Block2b(n_hidden=64)
        self.block_4 = Block2b(n_hidden=32)


    def forward(self, x):
        h = self.relu(self.fc1(x.reshape((-1, 3600))))
        h = self.relu(self.fc2(h))
        h = h.reshape((-1, 1, 60, 60))
        h = h - self.block_1(x, h)
        h = h - self.block_2(x, h)
        h = h - self.block_3(x, h)
        h = h - self.block_4(x, h)
        return h.reshape((-1, 60, 60))

class BorderPad_h(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        #first channel must be h
        super().__init__()
        self.n_pad = kernel_size//2
        self.refl_pad = nn.ReflectionPad2d(self.n_pad)

    def forward(self, x):
        z = self.refl_pad(x)
        z[:, 0, :, :self.n_pad//2] = (100-146)/37 - z[:, 0, :, :self.n_pad//2]


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

class Model13(nn.Module):
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


class Model14(nn.Module):
    #does not train
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(1, 3600*3, bias=False)
        self.conv0 = nn.Conv2d(1, 3, 1, padding='same', padding_mode='reflect')
        self.conv1 = nn.Conv2d(6, 8, 11, padding='same', padding_mode='reflect')
        self.conv2 = nn.Conv2d(8, 8, 17, padding='same', padding_mode='reflect')
        self.conv3 = nn.Conv2d(8, 8, 19, padding='same', padding_mode='reflect')
        self.conv4 = nn.Conv2d(8, 12, 17, padding='same', padding_mode='reflect')
        self.conv5 = nn.Conv2d(12, 16, 11, padding='same', padding_mode='reflect')
        self.conv6 = nn.Conv2d(16, 16, 7, padding='same', padding_mode='reflect')
        self.conv7 = nn.Conv2d(16, 10, 13, padding='same', padding_mode='reflect')
        self.conv8 = nn.Conv2d(10, 8, 13, padding='same', padding_mode='reflect')
        self.conv9 = nn.Conv2d(8, 4, 11, padding='same', padding_mode='reflect')
        self.conv10 = nn.Conv2d(4, 1, 9, padding='same', padding_mode='reflect')

        #self.pad5 = nn.ReflectionPad2d(5)
        #self.pad7 = nn.ReflectionPad2d(7)
        #self.pad9 = nn.ReflectionPad2d(9)
        #self.pad11 = nn.ReflectionPad2d(11)
        #self.pad15 = nn.ReflectionPad2d(15)

    def forward(self, x):
        h0 = torch.empty((x.shape[0], 6, 60, 60), device=x.device)
        h0[:, :3, :, :] = self.conv0(x)
        h0[:, 3:, :, :] = self.fc0(torch.ones((1, 1), device=x.device)).view(1, 3, 60, 60)
        h1 = self.relu(self.conv1(h0))
        h1 = self.relu(self.conv2(h1))
        h1 = self.relu(self.conv3(h1))
        h1 = self.relu(self.conv4(h1))
        h1 = self.relu(self.conv5(h1))
        h1 = self.relu(self.conv6(h1))
        h1 = self.relu(self.conv7(h1))
        h1 = self.relu(self.conv8(h1))
        h1 = self.relu(self.conv9(h1))
        h1 = self.relu(self.conv10(h1))
        return h1.view(-1, 60, 60)

#test fc after conv

if __name__ == "__main__":
    train_file_ids = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000", "_7000to8000"]
    #train_file_ids = ["0"]
    test_file_ids = ["_1000to1050", "_1050to1400"]

    x = torch.tensor(load_files("datasets/k_set", train_file_ids).reshape((-1, 1, 60, 60)), dtype=torch.float)
    z = torch.log(x)-4
    y = (torch.tensor(load_files("datasets/h_set", train_file_ids).reshape((-1, 60, 60)), dtype=torch.float)-146) / 37

    x_test = torch.tensor(load_files("datasets/k_set", test_file_ids).reshape((-1, 1, 60, 60)), dtype=torch.float)
    z_test = torch.log(x_test) - 4
    y_test = (torch.tensor(load_files("datasets/h_set", test_file_ids).reshape((-1, 60, 60)), dtype=torch.float) - 146) / 37


    print(torch.mean((y - torch.mean(y, dim=0))**2))
    print(torch.mean(y))

    # -------------------------
    # Device (GPU if available)
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # Move full datasets to device once (fast + simple; fits easily in VRAM here)
    z = z.to(device)
    y = y.to(device)
    z_test = z_test.to(device)
    y_test = y_test.to(device)

    # Checkpoint folder
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)

    # -------------------------
    # Model / loss / optimizer
    # -------------------------
    model_id = "12"
    model = Model12().to(device)
    print([i.numel() for i in model.parameters()], sum([i.numel() for i in model.parameters()]))

    # continue form existing model
    last = torch.load(out_dir / f"model{model_id}_last.pt", map_location=device)
    model.load_state_dict(last["model_state_dict"])

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=8e-6)

    # Baseline (predict mean field from train set)
    baseline = loss_fn(torch.mean(y, dim=0, keepdim=True), y_test).item()
    print("baseline loss:", baseline)

    # -------------------------
    # Training loop
    # -------------------------
    n_epochs = 360
    batch_size = 16
    batch_idx = np.arange(z.shape[0])

    best_test = float("inf")

    train_losses = []
    test_losses = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        np.random.shuffle(batch_idx)
        epoch_losses = []

        n_batches = (z.shape[0] - 1) // batch_size + 1
        for i in range(n_batches):
            idx = batch_idx[i * batch_size:(i + 1) * batch_size]

            pred = model(z[idx])
            loss = loss_fn(pred, y[idx])

            loss.backward()
            optim.step()
            optim.zero_grad()

            #print(epoch, i, loss.item())
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            pred_test = model(z_test)
            test_loss = loss_fn(pred_test, y_test).item()

        dt = time.time() - t0
        print(f"epoch {epoch}/{n_epochs} | train_loss {train_loss:.6f} | test_loss {test_loss:.6f} | {dt:.1f}s")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # -------------------------
        # Save checkpoints (last + best)
        # -------------------------
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "baseline_loss": baseline,
            # Normalization constants used in this code:
            "norm": {"logk_center": 4.0, "h_mean": 145.3243, "h_std": 35.5957},
            # Helpful metadata:
            "train_file_ids": train_file_ids,
            "test_file_ids": test_file_ids,
        }

        torch.save(ckpt, out_dir / f"model{model_id}_last.pt")
        if test_loss < best_test:
            best_test = test_loss
            torch.save(ckpt, out_dir / f"model{model_id}_best.pt")

    # -------------------------
    # Save predictions
    # -------------------------
    best = torch.load(out_dir / f"model{model_id}_best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])

    model.eval()
    with torch.no_grad():
        pred_test = model(z_test).detach().cpu().numpy()

    np.savetxt(f"pred_test{model_id}.txt", pred_test.reshape((-1, 3600)))

    print("saved:", f"pred_train{model_id}.txt, pred_test{model_id}.txt, checkpoints/model{model_id}_last.pt, checkpoints/model{model_id}_best.pt")

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["train_loss", "test_loss"])
    print("showing convergence")
    plt.savefig(f"convergence_plot{model_id}.png")
    plt.show()
