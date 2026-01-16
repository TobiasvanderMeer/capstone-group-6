import torch
from torch import nn

train_mode = 'default'
epochs = 5
lr = 8e-6

class Model(nn.Module):
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
