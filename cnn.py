import numpy as np
import torch
from torch import nn
from utils import load_files

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

    model = Model7()
    #print([i.numel() for i in model.parameters()])
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)

    print("baseline loss: ", loss_fn(torch.mean(y, dim=0, keepdim=True), y_test))

    n_epochs = 12  # test more epochs with convergence plot (with less data?)
    batch_size = 16
    batch_idx = np.arange(x.shape[0])
    for epoch in range(n_epochs):
        np.random.shuffle(batch_idx)
        b_losses = np.zeros((z.shape[0] - 1) // batch_size + 1)
        for i in range((z.shape[0] - 1) // batch_size + 1):
            pred = model.forward(z[batch_idx[i*batch_size:(i+1)*batch_size]])
            loss = loss_fn(pred, y[batch_idx[i*batch_size:(i+1)*batch_size]])
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

    np.savetxt("pred_train7b.txt", pred.detach().numpy().reshape((-1, 3600)))
    pred_test = model(z_test).detach().numpy()
    np.savetxt("pred_test7b.txt", pred_test.reshape((-1, 3600)))
