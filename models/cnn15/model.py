import torch
from torch import nn

train_mode = 'default'
epochs = 5
lr = 8e-6

class Model(nn.Module):
    #very bad trained at lr 1e-4
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 8, 11, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(8, 16, 17, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv2d(16, 32, 19, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(32, 48, 17, padding='same', padding_mode='reflect')
        self.conv5 = nn.Conv2d(48, 64, 11, padding='same', padding_mode='reflect')
        self.conv6 = nn.Conv2d(64, 64, 7, padding='same', padding_mode='zeros')
        self.conv7 = nn.Conv2d(64, 48, 13, padding='same', padding_mode='reflect')
        self.conv8 = nn.Conv2d(48, 32, 13, padding='same', padding_mode='reflect')
        self.conv9 = nn.Conv2d(32, 16, 11, padding='same', padding_mode='reflect')
        self.conv10 = nn.Conv2d(16, 1, 9, padding='same', padding_mode='reflect')

    def forward(self, x):
        h1 = self.relu(self.conv1(x))
        h1 = self.relu(self.conv2(h1))
        h1 = self.relu(self.conv3(h1))
        h2 = self.relu(self.conv4(h1))
        h2 = self.relu(self.conv5(h2))
        h2 = self.relu(self.conv6(h2))
        h2 = self.relu(self.conv7(h2))
        h2 = self.relu(self.conv8(h2)) + h1
        h2 = self.relu(self.conv9(h2))
        h2 = self.relu(self.conv10(h2))
        return h2.view(-1, 60, 60)

def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    print("Custom training not supported")
    return