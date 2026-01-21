import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import utils
import matplotlib.pyplot as plt
import importlib

#the code is this file is used to train all the models. Please be careful when modifying this code to ensure backwards
#compatibility


model_id = "cnn_fc2"  # change this to change different models

CONTINUE_FROM_LAST = False  # continue training from a previously saved model

model_file = importlib.import_module(f"models.{model_id}.model")  # this line imports the right model and training settings

#TODO:  save printed output in a text file in the folder

def default_train(n_epochs, lr, postfix):
    x = torch.tensor(utils.load_x_train().reshape((-1, 1, 60, 60)), dtype=torch.float)
    z = torch.log(x)-4
    y = (torch.tensor(utils.load_y_train().reshape((-1, 60, 60)), dtype=torch.float)-146) / 37

    x_test = torch.tensor(utils.load_x_test().reshape((-1, 1, 60, 60)), dtype=torch.float)
    z_test = torch.log(x_test) - 4
    y_test = (torch.tensor(utils.load_y_test().reshape((-1, 60, 60)), dtype=torch.float) - 146) / 37


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
    if CONTINUE_FROM_LAST:
        print("Continueing training from a previously trained model")
        # this code allows you to continue training on a model that you trained before. It will take the model_last.pt
        # (with postfix) and initialise the model with the parameters stored in this file. This is useful for example
        # when see you need more epochs or when you want to change training parameters during training but not reset
        # the training. Or when your computer crashes during training. (this works because the models saves itself after
        # every epoch)

        last = torch.load(out_dir / f"model_last{postfix}.pt", map_location=device)
        model.load_state_dict(last["model_state_dict"])
        # we also want to keep the convergence plot complete
        train_losses = last["train_loss_history"]
        test_losses = last["test_loss_history"]
        # we include the time of the previous training runs in the total training time
        total_train_time = last["training_time"]
        best_test_loss = min(test_losses)
    else:
        train_losses = []
        test_losses = []
        total_train_time = 0
        best_test_loss = float("inf")

    loss_fn = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Baseline (predict mean field from train set)
    baseline = loss_fn(torch.mean(y, dim=0, keepdim=True), y_test).item()
    print("baseline loss:", baseline)

    # -------------------------
    # Training loop
    # -------------------------
    batch_size = 16
    batch_idx = np.arange(z.shape[0])  # this one is used to shuffle the dataset

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()  # time how long each epoch takes
        model.train()
        np.random.shuffle(batch_idx)  # causes the dataset to come in a random order each epoch

        epoch_losses = []  # compute the training loss at each epoch by taking the mean of the losses of each batch

        n_batches = (z.shape[0] - 1) // batch_size + 1
        for i in range(n_batches):
            idx = batch_idx[i * batch_size:(i + 1) * batch_size]  # these are the indices of the samples in the batch

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
            # compute the test loss
            pred_test = model(z_test)
            test_loss = loss_fn(pred_test, y_test).item()

        dt = time.time() - t0  # how long this epoch took
        total_train_time += dt
        print(f"epoch {epoch}/{n_epochs} | train_loss {train_loss:.6f} | test_loss {test_loss:.6f} | {dt:.1f}s")
        # save the losses to generate a convergence plot
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # -------------------------
        # Save checkpoints (last + best)
        # -------------------------
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "train_loss_history": train_losses,
            "test_loss_history": test_losses,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "baseline_loss": baseline,
            "training_time": total_train_time,
            # Normalization constants used in this code:
            "norm": {"logk_center": 4.0, "h_mean": 146, "h_std": 37},  # TODO this is not equal for all models
            # Helpful metadata:
            "train_file_ids": utils.train_file_ids,
            "test_file_ids": utils.test_file_ids,
        }

        torch.save(ckpt, out_dir / f"model_last{postfix}.pt")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(ckpt, out_dir / f"model_best{postfix}.pt")

    print(f"finished training. Total training time: {total_train_time:.1f} seconds")
    # -------------------------
    # Save predictions
    # -------------------------
    best = torch.load(out_dir / f"model_best{postfix}.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])

    model.eval()
    with torch.no_grad():
        pred_test = model(z_test).detach().cpu().numpy()

    # the models predictions on the test set (used for analysis)
    np.savetxt(out_dir / f"pred_test{postfix}.txt", pred_test.reshape((-1, 3600)))

    print("saved:", f"pred_test{postfix}.txt, model_last{postfix}.pt, model_best.pt{postfix}")

    # plot the convergence
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["train_loss", "test_loss"])
    print("showing convergence")
    plt.savefig(out_dir /  f"convergence_plot{postfix}.png")
    plt.show()

if __name__ == "__main__":
    #each model needs a variable called "training mode" this variable determines what code is used to train the model
    train_mode = model_file.train_mode
    match train_mode:
        case 'default':
            #this training mode uses the variables "lr" and "epochs". This on should not be used and is only included for
            #backwards compatibility
            print("Using default training code to train", model_id)
            print("please use default2 for new models because that one is just better")
            default_train(model_file.epochs, model_file.lr, "")

        case 'default2':
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

        case 'custom':
            #for training algorithms that are specific to one model, please add the training code in the model file.
            #this one is not used (yet) so you can change the code without worrying about backwards compatibility too much
            model_file.custom_train()
        case _:
            print(f"{train_mode} is an invalid training mode")
