import torch
from torch import nn

train_mode = 'default'
epochs = 5
lr = 8e-6

class Model(nn.Module):
    #trained very poorly at 60 epochs lr 1e-6 and 60 at lr 1e-5
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 6, 11, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(6, 8, 17, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv2d(8, 8, 19, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(8, 12, 17, padding='same', padding_mode='reflect')
        self.conv5 = nn.Conv2d(12, 16, 11, padding='same', padding_mode='reflect')
        self.conv6 = nn.Conv2d(16, 16, 7, padding='same', padding_mode='zeros')
        self.conv7 = nn.Conv2d(16, 10, 13, padding='same', padding_mode='reflect')
        self.conv8 = nn.Conv2d(10, 8, 13, padding='same', padding_mode='reflect')
        self.conv9 = nn.Conv2d(8, 4, 11, padding='same', padding_mode='reflect')
        self.conv10 = nn.Conv2d(4, 1, 9, padding='same', padding_mode='reflect')

    def forward(self, x):
        h1 = self.relu(self.conv1(x))
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

def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    print("Custom training not supported")
    return