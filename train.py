import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils import load_files
import matplotlib.pyplot as plt
import importlib

#the code is this file is used to train all the models. Please be careful when modifying this code to ensure backwards
#compatibility

model_id = "cnn12"  # change this to change different models

model_file = importlib.import_module(f"models.{model_id}.model")  # this line imports the right model and training settings

#TODO:  print total time that elapsed during training
#       save printed output in a text file in the folder
#       fix the convergence plot when you continue training from a saved model and generaly improve this feature

def default_train(n_epochs, lr, postfix):
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
    out_dir = Path("models") / model_id
    out_dir.mkdir(exist_ok=True)

    # -------------------------
    # Model / loss / optimizer
    # -------------------------

    model = model_file.Model().to(device)
    print([i.numel() for i in model.parameters()], sum([i.numel() for i in model.parameters()]))

    # continue form existing model
    #last = torch.load(out_dir / f"model_last.pt", map_location=device)
    #model.load_state_dict(last["model_state_dict"])

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Baseline (predict mean field from train set)
    baseline = loss_fn(torch.mean(y, dim=0, keepdim=True), y_test).item()
    print("baseline loss:", baseline)

    # -------------------------
    # Training loop
    # -------------------------
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
            "norm": {"logk_center": 4.0, "h_mean": 145.3243, "h_std": 35.5957},  # TODO this is not equal for all models
            # Helpful metadata:
            "train_file_ids": train_file_ids,
            "test_file_ids": test_file_ids,
        }

        torch.save(ckpt, out_dir / f"model_last{postfix}.pt")
        if test_loss < best_test:
            best_test = test_loss
            torch.save(ckpt, out_dir / f"model_best{postfix}.pt")

    # -------------------------
    # Save predictions
    # -------------------------
    best = torch.load(out_dir / f"model_best{postfix}.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])

    model.eval()
    with torch.no_grad():
        pred_test = model(z_test).detach().cpu().numpy()

    np.savetxt(out_dir / f"pred_test{postfix}.txt", pred_test.reshape((-1, 3600)))

    print("saved:", f"pred_test{postfix}.txt, model_last{postfix}.pt, model_best.pt{postfix}")

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["train_loss", "test_loss"])
    print("showing convergence")
    plt.savefig(out_dir /  f"convergence_plot{postfix}.png")
    plt.show()

if __name__ == "__main__":
    #each model needs a variable called "training mode" this variable determines wha code is used to train the model
    train_mode = model_file.train_mode
    if train_mode == 'default':
        #this training mode uses the variables "lr" and "epochs". This on should not be used and is only included for
        #backwards compatibility
        print("Using default training code to train", model_id)
        print("please use default2 instead because default is outdated")
        default_train(model_file.epochs, model_file.lr, "")

    elif train_mode == 'default2':
        #this training mode uses the dictionary "training_settings" to store the training parameters.
        #this allows optional settings.
        print("Using default2 training code to train", model_id)
        epochs = model_file.training_settings["epochs"]
        lr = model_file.training_settings["lr"]
        if "postfix" in model_file.training_settings:
            postfix = model_file.training_settings["postfix"]
        else:
            postfix = ""
        print(f"lr: {lr}, n_epochs: {epochs}, postfix: \"{postfix}\"")
        default_train(epochs, lr, postfix)

    elif train_mode == 'custom':
        #for training algorithms that are specific to one model, please add the training code in the model file.
        #this one is not used (yet) so you can change the code without worrying about backwards compatibility too much
        model_file.custom_train()
    else:
        print(f"{train_mode} is an invalid training mode")
