import numpy as np
import scipy
from scipy.sparse.linalg import LaplacianNd, spsolve
from scipy.sparse import csr_matrix

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

n = 64                  # grid size (64x64)
L = 6.0                 # domain length (km)
dx = L / (n - 1)        # grid spacing
NSAMPLES = 10000       # total number of samples
BATCH_SIZE = 1000      # samples per save batch


# ============================================================
# HYDRAULIC CONDUCTIVITY FIELD
# ============================================================

def hydraulic_conductivity_field(n, seed):
    np.random.seed(seed)

    u_bar = 4.0
    beta = 0.5
    alpha = 1.3

    lap = LaplacianNd((n, n), boundary_conditions="neumann")

    eigvecs = lap.eigenvectors()[:, :-1]
    eigvals = (-lap.eigenvalues()[:-1]) ** (-alpha / 2)

    u = (
        beta**0.5
        * eigvecs
        @ np.diag(eigvals)
        @ eigvecs.T
        @ np.random.randn(n**2)
        + u_bar
    )

    return np.exp(u)


# ============================================================
# SOURCE FUNCTION
# ============================================================

def source_function(n):
    f = np.zeros(n**2)

    def source(x2):
        if 0 <= x2 <= 4:
            return 0.0
        elif 4 < x2 < 5:
            return 137.0
        else:
            return 274.0

    for i in range(n):
        for j in range(n):
            idx = j * n + i
            x2 = j * dx
            f[idx] = source(x2)

    return f


# ============================================================
# DARCY SOLVER
# ============================================================

def solve_darcy_flow(n, kappa, f):
    b = f * dx**2

    def idx(i, j):
        return j * n + i

    def k_face(i1, j1, i2, j2):
        k1 = kappa[idx(i1, j1)]
        k2 = kappa[idx(i2, j2)]
        return 2 * k1 * k2 / (k1 + k2)

    A = np.zeros((n**2, n**2))

    for i in range(n):
        for j in range(n):
            k = idx(i, j)

            # Interior
            if 0 < i < n - 1 and 0 < j < n - 1:
                kE = k_face(i, j, i + 1, j)
                kW = k_face(i, j, i - 1, j)
                kN = k_face(i, j, i, j + 1)
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kE + kW + kN + kS
                A[k, k + 1] = -kE
                A[k, k - 1] = -kW
                A[k, k + n] = -kN
                A[k, k - n] = -kS

            # Left boundary (flux)
            elif i == 0 and 0 < j < n - 1:
                kE = k_face(i, j, i + 1, j)
                kN = k_face(i, j, i, j + 1)
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kE + kN + kS
                A[k, k + 1] = -kE
                A[k, k + n] = -kN
                A[k, k - n] = -kS
                b[k] += 500 * dx

            # Right boundary (zero gradient)
            elif i == n - 1 and 0 < j < n - 1:
                kW = k_face(i, j, i - 1, j)
                kN = k_face(i, j, i, j + 1)
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kW + kN + kS
                A[k, k - 1] = -kW
                A[k, k + n] = -kN
                A[k, k - n] = -kS

            # Bottom boundary (Dirichlet)
            elif j == 0:
                A[k, k] = 1.0
                b[k] = 100.0

            # Top boundary (zero gradient)
            elif j == n - 1:
                kE = k_face(i, j, i + 1, j) if i < n - 1 else 0
                kW = k_face(i, j, i - 1, j) if i > 0 else 0
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kE + kW + kS
                if i < n - 1:
                    A[k, k + 1] = -kE
                if i > 0:
                    A[k, k - 1] = -kW
                A[k, k - n] = -kS

    return spsolve(csr_matrix(A), b)


# ============================================================
# BATCH DATA GENERATION + SAVE
# ============================================================

def generate_and_save_data(seeds):
    f = source_function(n)

    X_batch = []
    Y_batch = []
    batch_id = 0

    for i, seed in enumerate(seeds):
        print(f"Sample {i+1}/{len(seeds)} | seed={seed}")

        kappa = hydraulic_conductivity_field(n, seed)
        h = solve_darcy_flow(n, kappa, f)

        X_batch.append(kappa)
        Y_batch.append(h)

        if (i + 1) % BATCH_SIZE == 0:
            save_batch(X_batch, Y_batch, batch_id)
            X_batch, Y_batch = [], []
            batch_id += 1

    # Save remaining samples
    if len(X_batch) > 0:
        save_batch(X_batch, Y_batch, batch_id)


def save_batch(X, Y, batch_id):
    X = np.array(X)
    Y = np.array(Y)

    np.savetxt(f"datasets/k_set_64x64_batch{batch_id}.txt", X)
    np.savetxt(f"datasets/h_set_64x64_batch{batch_id}.txt", Y)

    print(f"Saved batch {batch_id}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    seeds = range(NSAMPLES)
    generate_and_save_data(seeds)
    print("All data generated and saved.")
