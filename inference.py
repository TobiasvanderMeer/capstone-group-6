import numpy as np
import matplotlib.pyplot as plt
import torch
import importlib
from pathlib import Path

np.random.seed(42)

from jeffrey_code import hydraulic_conductivity_field, source_function

class Predictor:
    def __init__(self, model_id, postfix, n):

        model_file = importlib.import_module(f"models.{model_id}.model")  # this line imports the right model and training settings

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Checkpoint folder
        out_dir = Path("models") / model_id
        out_dir.mkdir(exist_ok=True)

        # Load model checkpoint
        ckpt_path = out_dir / f"model_last{postfix}.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.n = n

        self.model = model_file.Model().to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.logk_center = ckpt["norm"]["logk_center"]
        self.h_mean = ckpt["norm"]["h_mean"]
        self.h_std = ckpt["norm"]["h_std"]

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
    kappa = np.empty((BATCH_SIZE, 3600))
    for j, seed in enumerate(seeds):
        kappa[j] = hydraulic_conductivity_field(n, seed)
    return kappa

def main():
    # make a random input
    kappa = random_input()

    # make the predictor
    predictor = Predictor("cnn12c", "_lr6e-5", 60)
    # this is the inference
    h_pred = predictor.predict(kappa)[0]

    # show results
    plt.imshow(h_pred, cmap='hot_r', origin='lower', interpolation='none')
    plt.title(f'Predicted h')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
