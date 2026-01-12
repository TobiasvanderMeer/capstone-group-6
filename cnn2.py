import numpy as np
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


def load_files(filename: str, file_ids: list[str]):
    return np.concatenate([np.loadtxt(filename + file_id + ".txt") for file_id in file_ids])


if __name__ == "__main__":
    train_file_ids = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000", "_7000to8000"]
    test_file_ids = ["_1000to1050", "_1050to1400"]

    x = torch.tensor(load_files("datasets/k_set", train_file_ids).reshape((-1, 1, 60, 60)), dtype=torch.float)
    z = torch.log(x)-4
    y = (torch.tensor(load_files("datasets/h_set", train_file_ids).reshape((-1, 60, 60)), dtype=torch.float)-146) / 37

    x_test = torch.tensor(load_files("datasets/k_set", test_file_ids).reshape((-1, 1, 60, 60)), dtype=torch.float)
    z_test = torch.log(x_test) - 4
    y_test = (torch.tensor(load_files("datasets/h_set", test_file_ids).reshape((-1, 60, 60)), dtype=torch.float) - 146) / 37


    print(torch.mean((y - torch.mean(y, dim=0))**2))
    print(torch.mean(y))

    model = Model()
    # print([i.numel() for i in model.parameters()])
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("baseline loss: ", loss_fn(torch.mean(y, dim=0, keepdim=True), y_test))

    n_epochs = 8
    batch_size = 50
    batch_idx = np.arange(x.shape[0])
    for epoch in range(n_epochs):
        r = torch.randn(z.shape)
        print(r.shape)
        t = torch.rand((z.shape[0]))
        w = r*t.view(-1, 1, 1, 1)
        zz = w + z*(1-t).view(-1, 1, 1, 1)

        np.random.shuffle(batch_idx)
        b_losses = np.zeros((z.shape[0] - 1) // batch_size + 1)
        for i in range((z.shape[0] - 1) // batch_size + 1):
            batch = batch_idx[i * batch_size:(i + 1) * batch_size]
            pred = model.forward(z[batch], zz[batch], t[batch])
            loss = loss_fn(pred, w[batch])
            loss.backward()
            optim.step()
            optim.zero_grad()
            b_losses[i] = loss.item()
            print(loss.item())

        print(epoch, np.mean(b_losses))
        with torch.no_grad():
            pred_test = model(z_test)
            test_loss = loss_fn(pred_test, y_test)
            print("test_loss", test_loss.item())
