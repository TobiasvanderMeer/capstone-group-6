import torch
from torch import nn

train_mode = 'default'
epochs = 5
lr = 8e-6

class Model(nn.Module):
    # her I added residual connection to improve training. The model converges faster than model15 but still very bad
    # performance. Also takes much longer to train (48 seconds per epoch)
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 8, 11, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(8, 16, 11, padding='same', padding_mode='zeros')

        self.conv3 = nn.Conv2d(16, 16, 19, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(16, 16, 19, padding='same', padding_mode='reflect')
        self.conv5 = nn.Conv2d(16, 16, 19, padding='same', padding_mode='reflect')

        self.conv6 = nn.Conv2d(16, 16, 13, padding='same', padding_mode='zeros')
        self.conv7 = nn.Conv2d(16, 16, 13, padding='same', padding_mode='reflect')
        self.conv8 = nn.Conv2d(16, 16, 13, padding='same', padding_mode='reflect')

        self.conv9 = nn.Conv2d(16, 16, 9, padding='same', padding_mode='zeros')
        self.conv10 = nn.Conv2d(16, 16, 9, padding='same', padding_mode='zeros')
        self.conv11 = nn.Conv2d(16, 16, 9, padding='same', padding_mode='zeros')

        self.conv12 = nn.Conv2d(16, 1, 7, padding='same', padding_mode='reflect')

    def forward(self, x):
        h1 = self.relu(self.conv1(x))
        h1 = self.relu(self.conv2(h1))

        h2 = self.relu(self.conv3(h1))
        h2 = self.relu(self.conv4(h2))
        h2 = self.relu(self.conv5(h2)) + h1

        h3 = self.relu(self.conv6(h2))
        h3 = self.relu(self.conv7(h3))
        h3 = self.relu(self.conv8(h3)) + h2

        h4 = self.relu(self.conv9(h3))
        h4 = self.relu(self.conv10(h4))
        h4 = self.relu(self.conv11(h4)) + h3

        h5 = self.relu(self.conv12(h4))
        return h5.view(-1, 60, 60)
