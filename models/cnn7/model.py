import torch
from torch import nn

train_mode = 'default'
epochs = 50
lr = 8e-2

class Model(nn.Module):
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

def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    print("Custom training not supported")
    return