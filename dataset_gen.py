from jeffrey_code import hydraulic_conductivity_field, solve_darcy_flow, source_function
import numpy as np

n = 64

def generate_data(seeds):
    x = np.empty((len(seeds), n**2))
    y = np.empty((len(seeds), n**2))

    for i, seed in enumerate(seeds):
        print(i)
        x[i, :] = hydraulic_conductivity_field(n, seed)
        f = source_function(n)
        y[i, :] = solve_darcy_flow(n, x[i, :], f)
    return x, y

def save_data(x, y, id=""):
    np.savetxt(f"datasets/k_set{id}.txt", x.reshape(-1, n*n))
    np.savetxt(f"datasets/h_set{id}.txt", y.reshape(-1, n*n))

if __name__ == "__main__":
    start_seed, stop_seed = 0, 1000
    print(f"computing seeds {start_seed} to {stop_seed}")
    x, y =generate_data(list(range(start_seed, stop_seed)))
    save_data(x, y, id=f"_{start_seed}to{stop_seed}")