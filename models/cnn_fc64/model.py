import torch
from torch import nn

train_mode = 'default2'

training_settings = {"epochs": 400,
                     "lr": 2e-5,
                     "postfix": "_b50",
                     "batch_size": 50,
                     "size": 64}

class Model(nn.Module):
    # This is a version of cnn_fc but now 64x64

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 16, 5, padding='same', padding_mode='reflect')
        self.conv2 = nn.Conv2d(16, 16, 7, padding='same', padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(2)  # 32x32
        self.conv3 = nn.Conv2d(16, 32, 5, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(32, 32, 5, padding='same', padding_mode='zeros')
        self.pool2 = nn.MaxPool2d(2)  # 16x16
        self.conv5 = nn.Conv2d(32, 64, 5, padding='same', padding_mode='zeros')
        self.conv6 = nn.Conv2d(64, 64, 5, padding='same', padding_mode='zeros')
        self.pool3 = nn.MaxPool2d(2) # 8x8
        self.conv7 = nn.Conv2d(64, 64, 5, padding='same', padding_mode='zeros')
        self.conv8 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='zeros')

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 4096)



    def forward(self, x):
        #convolutional part
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.pool1(h)

        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = self.pool2(h)

        h = self.relu(self.conv5(h))
        h = self.relu(self.conv6(h))
        h = self.pool3(h)

        h = self.relu(self.conv7(h))
        h = self.relu(self.conv8(h)).view(-1, 4096)

        # fully connected part
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.fc3(h)

        return h.reshape((-1, 64, 64))

def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    print("Custom training not supported")
    return