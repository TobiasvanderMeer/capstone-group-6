import numpy as np
import matplotlib.pyplot as plt
from utils import load_files
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3600, 3600)
        self.fc2 = nn.Linear(3600, 3600)
        self.conv1 = nn.Conv2d(5, 8, 9, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv2d(8, 8, 9, padding='same', padding_mode='zeros')
        self.conv4 = nn.Conv2d(8, 1, 9, padding='same', padding_mode='zeros')

    def forward(self, x, hr, t):
        z = torch.empty((x.shape[0], 5, 60, 60))
        z[:, 0, :, :] = x.view(-1, 60, 60)
        z[:, 1, :, :] = self.relu(self.fc1(x.view(-1, 3600))).view(-1, 60, 60)
        z[:, 2, :, :] = hr.view(-1, 60, 60)
        z[:, 3, :, :] = self.relu(self.fc2(hr.view(-1, 3600)).view(-1, 60, 60))
        z[:, 4, :, :] = t.view(-1, 1, 1).expand(-1, 60, 60)

        r = self.relu(self.conv1(z))
        r = self.relu(self.conv2(r))
        r = self.relu(self.conv3(r))
        r = self.conv4(r)
        return r


set_ids = ["_1000to1050", "_1050to1400"]  # the set that was used as test set in the cnn file
#set_id = "0"
#set_id = "_test"
result_id = "_test"  # we want to look at the performance on the test set

x = load_files("datasets/k_set", set_ids).reshape((-1, 60, 60))
xt = torch.tensor(x, dtype=torch.float)
z = torch.log(xt)-4
y = load_files("datasets/h_set", set_ids).reshape((-1, 60, 60))
yt = (torch.tensor(y, dtype=torch.float)-146)/37
#pred = np.loadtxt(f"pred{result_id}.txt").reshape((-1, 60, 60))*37+146

model = torch.load('cnn2.pt', weights_only=False)

for i in range(100):
    t = torch.tensor(0.5)
    r = torch.randn((1, 1, 60, 60))
    w = r * torch.sqrt(t).view(-1, 1, 1, 1)
    zz = w + yt.view(-1, 1, 60, 60) * (1 - t).view(-1, 1, 1, 1)

    pred = (zz - model(z, zz, t).detach().numpy())[0, 0]*37+146
    #print("MAE: ", np.mean(np.abs(y[i] - pred[i])))
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(np.log(x[i].transpose()))
    # plt.ion()
    ax2.imshow(y[i], interpolation='none')
    #ax2.contour(y[i], levels=20, colors=["black"])
    ax3.imshow(pred, interpolation='none')
    #ax3.contour(pred, levels=20, colors=["black"])
    ax4.imshow(zz[0, 0], interpolation='none')
    plt.show(block=True)