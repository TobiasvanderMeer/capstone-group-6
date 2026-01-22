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


model_id = "cnn_fc64"  # change this to change different models

CONTINUE_FROM_LAST = False  # continue training from a previously saved model

model_file = importlib.import_module(f"models.{model_id}.model")  # this line imports the right model and training settings

#TODO:  save printed output in a text file in the folder

def default_train(n_epochs, lr, postfix, batch_size=16, n=60):
    if n == 60:
        x = torch.tensor(utils.load_x_train().reshape((-1, 1, n, n)), dtype=torch.float)
        y = (torch.tensor(utils.load_y_train().reshape((-1, n, n)), dtype=torch.float)-146) / 37

        x_test = torch.tensor(utils.load_x_test().reshape((-1, 1, n, n)), dtype=torch.float)
        y_test = (torch.tensor(utils.load_y_test().reshape((-1, n, n)), dtype=torch.float) - 146) / 37
    if n == 64:
        x = torch.tensor(utils.load_x_train64().reshape((-1, 1, n, n)), dtype=torch.float)
        y = (torch.tensor(utils.load_y_train64().reshape((-1, n, n)), dtype=torch.float) - 146) / 37

        x_test = torch.tensor(utils.load_x_test64().reshape((-1, 1, n, n)), dtype=torch.float)
        y_test = (torch.tensor(utils.load_y_test64().reshape((-1, n, n)), dtype=torch.float) - 146) / 37

    z = torch.log(x)-4
    z_test = torch.log(x_test) - 4

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
    np.savetxt(out_dir / f"pred_test{postfix}.txt", pred_test.reshape((-1, n*n)))

    print("saved:", f"pred_test{postfix}.txt, model_last{postfix}.pt, model_best.pt{postfix}")

    # plot the convergence
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["train_loss", "test_loss"])
    print("showing convergence")
    plt.savefig(out_dir /  f"convergence_plot{postfix}.png")
    plt.show()


def unet_train(base_ch: int = 64, ENFORCE_DIRICHLET_ROW0: bool = True):
    # ============================================================
    # Config
    # ============================================================

    # Batch files: datasets/k_set_64x64_batch{i}.txt
    TRAIN_BATCH_IDS = list(range(0, 6))
    TEST_BATCH_IDS = list(range(6, 8))

    LOGK_CENTER = 4.0
    H_MEAN = 145.3243
    H_STD = 35.5957

    N_EPOCHS = 80
    BATCH_SIZE = 16

    BASE_CH = 64
    LR = 3e-4
    WEIGHT_DECAY = 1e-5

    # Early stopping
    PATIENCE = 12
    MIN_DELTA = 1e-4

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4

    # Prediction batching
    PRED_BATCH_TRAIN = 16
    PRED_BATCH_TEST = 32

    SEED = 0

    # ============================================================
    # Data loading
    # ============================================================

    def load_batches(prefix: str, batch_ids: list[int]) -> np.ndarray:
        parts = []
        for i in batch_ids:
            fname = f"datasets/{prefix}_batch{i}.txt"
            print("loading", fname)
            parts.append(np.loadtxt(fname, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def predict_in_batches(model: nn.Module, z: torch.Tensor, batch_size: int) -> np.ndarray:
        model.eval()
        n = z.shape[0]
        out = np.empty((n, 64, 64), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                zb = z[i:i + batch_size]
                pb = model(zb).squeeze(1).cpu().numpy()
                out[i:i + pb.shape[0]] = pb

        return out

    # ============================================================
    # Main
    # ============================================================


    torch.manual_seed(SEED)
    np.random.seed(SEED)

        # -------------------------
        # Load data
        # -------------------------
    x = load_batches("k_set_64x64", TRAIN_BATCH_IDS)
    y = load_batches("h_set_64x64", TRAIN_BATCH_IDS)

    x_test = load_batches("k_set_64x64", TEST_BATCH_IDS)
    y_test = load_batches("h_set_64x64", TEST_BATCH_IDS)

    x = torch.tensor(x.reshape((-1, 1, 64, 64)))
    y = torch.tensor(y.reshape((-1, 64, 64)))

    x_test = torch.tensor(x_test.reshape((-1, 1, 64, 64)))
    y_test = torch.tensor(y_test.reshape((-1, 64, 64)))

        # Normalize
    z = torch.log(x) - LOGK_CENTER
    y = (y - H_MEAN) / H_STD

    z_test = torch.log(x_test) - LOGK_CENTER
    y_test = (y_test - H_MEAN) / H_STD

        # -------------------------
        # Device
        # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("using device:", device)

    z = z.to(device)
    y = y.to(device)
    z_test = z_test.to(device)
    y_test = y_test.to(device)

        # -------------------------
        # Model
        # -------------------------
    model = model_file.Model(
        base_ch=BASE_CH,
        enforce_dirichlet_row0=ENFORCE_DIRICHLET_ROW0
    ).to(device)

    loss_fn = nn.MSELoss()

    optim = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    # Baseline (mean field)
    mean_field = torch.mean(y, dim=0)
    baseline = torch.mean((y_test - mean_field) ** 2).item()
    print("baseline loss:", baseline)

        # -------------------------
        # Checkpoints
        # -------------------------
    out_dir = Path(f"models/{model_id}")
    out_dir.mkdir(exist_ok=True)

    best_test = float("inf")
    best_epoch = 0
    bad_epochs = 0

    idx_all = np.arange(z.shape[0])

    train_losses = []
    test_losses = []
    total_train_time = 0
        # -------------------------
        # Training loop
        # -------------------------
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        np.random.shuffle(idx_all)

        losses = []
        n_batches = (z.shape[0] - 1) // BATCH_SIZE + 1

        for b in range(n_batches):
            idx = idx_all[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            pred = model(z[idx]).squeeze(1)
            loss = loss_fn(pred, y[idx])

            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            pred_test = model(z_test).squeeze(1)
            test_loss = loss_fn(pred_test, y_test).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        lr_before = optim.param_groups[0]["lr"]
        scheduler.step(test_loss)
        lr_after = optim.param_groups[0]["lr"]

        if lr_after < lr_before:
            print(f"lr reduced: {lr_before:.2e} → {lr_after:.2e}")

        dt = time.time() - t0
        total_train_time += dt
        print(
                f"epoch {epoch:03d}/{N_EPOCHS} | "
                f"lr {lr_after:.2e} | "
                f"train {train_loss:.6f} | "
                f"test {test_loss:.6f} | "
                f"{dt:.1f}s"
            )

            # Save last
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
                "norm": {"logk_center": LOGK_CENTER, "h_mean": H_MEAN, "h_std": H_STD},
                "model": {"name": "UNet64", "base_ch": BASE_CH},
            }
        torch.save(ckpt, out_dir / "model_last.pt")

            # Best + early stopping
        if test_loss < best_test - MIN_DELTA:
            best_test = test_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(ckpt, out_dir / "model_best.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"early stopping (best epoch {best_epoch}, loss {best_test:.6f})")
            break

        # -------------------------
        # Export predictions
        # -------------------------
    best = torch.load(out_dir / "model_best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred_test = predict_in_batches(model, z_test, PRED_BATCH_TEST)

    np.savetxt(out_dir / "pred_test.txt", pred_test.reshape((-1, 4096)))

    print("saved predictions + checkpoints")
    print(f"best test loss: {best_test:.6f} (epoch {best_epoch})")


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
            batch = model_file.training_settings["batch_size"] if "batch_size" in model_file.training_settings else 16
            size = model_file.training_settings["size"] if "size" in model_file.training_settings else 60
            print(f"lr: {lr}, n_epochs: {epochs}, postfix: \"{postfix}\", batch size: {batch}, n: {size}")
            default_train(epochs, lr, postfix, batch_size=batch, n=size)

        case 'unet':
            # this training mode uses the dictionary "training_settings" to store the training parameters.
            # this allows optional settings.
            print("Using unet training code to train", model_id)
            unet_train()

        case 'custom':
            #for training algorithms that are specific to one model, please add the training code in the model file.
            #this one is not used (yet) so you can change the code without worrying about backwards compatibility too much
            print("Using custom training code to train", model_id)
            model_file.custom_train()
        case _:
            print(f"{train_mode} is an invalid training mode")
