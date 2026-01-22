import numpy as np
import matplotlib.pyplot as plt
import torch
import importlib
from pathlib import Path

np.random.seed(42)

#this file can be imported to easily use our models for inference, with only two lines of code. see main() for an example of how to use this.

from jeffrey_code import hydraulic_conductivity_field, source_function

class Predictor:
    def __init__(self, model_id, postfix, n, use_gpu=True):

        model_file = importlib.import_module(f"models.{model_id}.model")  # this line imports the right model and training settings

        # Device
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        print("Using device:", self.device)

        # Checkpoint folder
        out_dir = Path("models") / model_id
        out_dir.mkdir(exist_ok=True)

        # Load model checkpoint
        ckpt_path = out_dir / f"model_last{postfix}.pt"
        self.ckpt = torch.load(ckpt_path, map_location=self.device)

        self.n = n

        self.model = model_file.Model().to(self.device)
        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.eval()

        self.logk_center = self.ckpt["norm"]["logk_center"]
        self.h_mean = self.ckpt["norm"]["h_mean"]
        self.h_std = self.ckpt["norm"]["h_std"]

    def predict(self, kappa):
        # Predicted hydraulic head
        x_input = torch.tensor(np.log(kappa) - self.logk_center, dtype=torch.float32).view((-1, 1, self.n, self.n)).to(self.device)

        with torch.no_grad():
            h_pred_norm = self.model(x_input).view(-1, self.n, self.n)
        return h_pred_norm.cpu().numpy() * self.h_std + self.h_mean


def random_input(n=60):
    BATCH_SIZE = 16
    seeds = np.random.choice(range(10000), size=BATCH_SIZE, replace=False)
    # Generate random hydraulic conductivity field; replace this with the input
    kappa = np.empty((BATCH_SIZE, n*n))
    for j, seed in enumerate(seeds):
        kappa[j] = hydraulic_conductivity_field(n, seed)
    return kappa

def main():
    model_id = "cnn_fc64"
    postfix = "_b16"
    n = 64
    # make a random input
    kappa = random_input(n)

    # make the predictor
    predictor = Predictor(model_id, postfix, n)
    # this is the inference
    h_pred = predictor.predict(kappa)[0]

    print("showing results")
    # show results
    plt.imshow(h_pred, cmap='hot_r', origin='lower', interpolation='none')
    plt.title(f'Predicted h')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
