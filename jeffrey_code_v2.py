## Import libraries

import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.sparse.linalg import LaplacianNd, spsolve


## parameters

n = 60              # number of grid points in each direction
L = 6               # length of the domain in each direction (km)
dx = L/(n-1)        # grid spacing (km)


#seed = 9         # seed for generating hydraulic conductivity field (seed = 3131 to reproduce Iglesias et al. (2013) truth)

def hydraulic_conductivity_field(n,seed):

    " Generate hydraulic conductivity field k on a n x n grid using a Gaussian random field u with given parameters. "

    # keep fixed
    u_bar = 4
    beta = 0.5
    alpha = 1.3
    np.random.seed(seed)

    grid_shape = (n,n)
    lap = LaplacianNd(grid_shape, boundary_conditions = 'neumann')

    u = beta**(1/2) * lap.eigenvectors()[:,:-1] @ np.diag((-lap.eigenvalues()[:-1])**(-alpha/2)) @ (lap.eigenvectors()[:,:-1]).T @ np.random.randn(n**2) + u_bar*np.ones(n**2)

    k = np.exp(u)

    return k

def plot_hydraulic_conductivity_field(n,seed):
    kappa = hydraulic_conductivity_field(n,seed)

    plt.imshow(np.log(kappa).reshape(n,n), cmap = 'jet', origin = 'lower')
    plt.colorbar()
    plt.xticks(np.linspace(0,n-1,7), labels = range(7))
    plt.yticks(np.linspace(0,n-1,7), labels = range(7))
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title('Logarithm of hydraulic conductivity field')
    plt.show()


## source function

def source_function(n):
    " Generate source function f on a n x n grid using given parameters. "

    L = 6
    dx = L / (n - 1)

    def source(x1, x2):
        if 0 <= x2 <= 4:
            return 0.0
        elif 4 < x2 < 5:
            return 137.0
        elif 5 <= x2 <= 6:
            return 274.0

    f = np.zeros(n ** 2)

    for i in range(n):
        for j in range(n):
            idx = j * n + i
            x1 = i * dx
            x2 = j * dx
            f[idx] = source(x1, x2)

    return f


def plot_source_function(n, seed):
    f = source_function(n)

    plt.imshow(f.reshape(n, n), origin='lower', cmap='jet')
    plt.colorbar()
    plt.xticks(np.linspace(0, n - 1, 7), labels=range(7))
    plt.yticks(np.linspace(0, n - 1, 7), labels=range(7))
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title('Recharge function f')
    plt.show()

    f = source_function(n)

    kappa = hydraulic_conductivity_field(n, seed)


def solve_darcy_flow(n, kappa, f):
    " Solve steady-state Darcy flow equation on a n x n grid with hydraulic conductivity field kappa and source function f. "

    L = 6  # length of the domain in each direction (km)
    dx = L / (n - 1)  # grid spacing (km)
    b = f * dx ** 2  # right-hand side vector of system Ax = b

    ## recover ID
    def idx(i, j):
        return j * n + i

    ## calculate kappa at faces using harmonic averaging
    def k_face(i1, j1, i2, j2):
        k1 = kappa[idx(i1, j1)]
        k2 = kappa[idx(i2, j2)]

        # harmonic average
        avg = 2 * k1 * k2 / (k1 + k2)

        return avg

    ## build discretization matrix A
    A = np.zeros((n ** 2, n ** 2))

    for i in range(n):
        for j in range(n):
            k = idx(i, j)

            ## interior points
            if 0 < i < n - 1 and 0 < j < n - 1:
                k_E = k_face(i, j, i + 1, j)  # k_{i+1/2,j}
                k_N = k_face(i, j, i, j + 1)  # k_{i,j+1/2}
                k_W = k_face(i, j, i - 1, j)  # k_{i-1/2,j}
                k_S = k_face(i, j, i, j - 1)  # k_{i,j-1/2}

                A[k, k] = k_N + k_E + k_S + k_W
                A[k, k + 1] = -k_E
                A[k, k - 1] = -k_W
                A[k, k + n] = -k_N
                A[k, k - n] = -k_S

            ## Left boundary: - k h_x(0,y) = 500
            if i == 0 and 0 < j < n - 1:
                k_E = k_face(i, j, i + 1, j)
                k_N = k_face(i, j, i, j + 1)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_N + k_E + k_S
                A[k, k + 1] = -k_E
                A[k, k + n] = -k_N
                A[k, k - n] = -k_S

                b[k] += 500 * dx

                ## Right boundary: h_x(6,y) = 0
            if i == n - 1 and 0 < j < n - 1:
                k_W = k_face(i, j, i - 1, j)
                k_N = k_face(i, j, i, j + 1)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_N + k_W + k_S
                A[k, k - 1] = -k_W
                A[k, k + n] = -k_N
                A[k, k - n] = -k_S

            ## Lower boundary: h(x,0) = 100
            if j == 0 and 0 < i < n - 1:
                A[k, k] = 1

                b[k] = 100

            ## Upper boundary: h_y(x,6) = 0
            if j == n - 1 and 0 < i < n - 1:
                k_E = k_face(i, j, i + 1, j)
                k_W = k_face(i, j, i - 1, j)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_E + k_W + k_S
                A[k, k + 1] = -k_E
                A[k, k - 1] = -k_W
                A[k, k - n] = -k_S

            ## Corners

            ## Lower left corner
            if i == 0 and j == 0:
                A[k, k] = 1
                b[k] = 100

            ## Lower right corner
            if i == n - 1 and j == 0:
                A[k, k] = 1
                b[k] = 100

            ## Upper left corner
            if i == 0 and j == n - 1:
                k_E = k_face(i, j, i + 1, j)
                k_S = k_face(i, j, i, j - 1)
                A[k, k] = k_E + k_S
                A[k, k + 1] = -k_E
                A[k, k - n] = -k_S

                b[k] += 500 * dx

            ## Upper right corner
            if i == n - 1 and j == n - 1:
                k_W = k_face(i, j, i - 1, j)
                k_S = k_face(i, j, i, j - 1)
                A[k, k] = k_W + k_S
                A[k, k - 1] = -k_W
                A[k, k - n] = -k_S

    # solve the sparse linear system Ax = b
    h = spsolve(scipy.sparse.csr_matrix(A), b)

    return h


def test_solve_darcy_flow(n, seed):
    kappa = hydraulic_conductivity_field(n, seed)
    f = source_function(n)
    h = solve_darcy_flow(n, kappa, f)

    fig, ax = plt.subplots()

    CS = ax.contour(h.reshape(n, n), levels=range(100, 400, 10), colors='k')
    ax.clabel(CS, fontsize=9)
    # plt.imshow(h.reshape(n,n), cmap = 'hot_r', origin = 'lower')
    # plt.colorbar()
    plt.xticks(np.linspace(0, n - 1, 7), labels=range(7))
    plt.yticks(np.linspace(0, n - 1, 7), labels=range(7))
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title('Steady-state hydraulic head distribution')
    plt.xlim(0, n - 1)
    plt.ylim(0, n - 1)
    plt.show()



## calculate gradients in x and y direction to add streamlines into the picture

def hydraulic_head_gradient(n, h, kappa):

    " Calculate hydraulic head gradients hx and hy on a n x n grid using central differences. "

    L = 6           # length of the domain in each direction (km)
    dx = L/(n-1)    # grid spacing (km)

    hx = np.zeros(n**2)
    hy = np.zeros(n**2)

    def idx(i,j):
        return j*n + i

    for i in range(n):
        for j in range(n):
            k = idx(i,j)

            # interior points
            if 0<i<n-1 and 0<j<n-1:
                hx[k] = (h[idx(i+1,j)] - h[idx(i-1,j)]) / (2*dx)
                hy[k] = (h[idx(i,j+1)] - h[idx(i,j-1)]) / (2*dx)

            # left boundary
            if i == 0 and 0 < j < n-1:
                hx[k] = (h[idx(i+1,j)] - h[idx(i,j)]) / dx
                hy[k] = (h[idx(i,j+1)] - h[idx(i,j-1)]) / (2*dx)

            # right boundary
            if i == n-1 and 0 < j < n-1:
                hx[k] = (h[idx(i,j)] - h[idx(i-1,j)]) / dx
                hy[k] = (h[idx(i,j+1)] - h[idx(i,j-1)]) / (2*dx)

            # lower boundary
            if j == 0 and 0 < i < n-1:
                hx[k] = (h[idx(i+1,j)] - h[idx(i-1,j)]) / (2*dx)
                hy[k] = (h[idx(i,j+1)] - h[idx(i,j)]) / dx

            # upper boundary
            if j == n-1 and 0 < i < n-1:
                hx[k] = (h[idx(i+1,j)] - h[idx(i-1,j)]) / (2*dx)
                hy[k] = (h[idx(i,j)] - h[idx(i,j-1)]) / dx

            # corners can be handled similarly if needed


        U = -kappa * hx
        V = -kappa * hy

    return U,V


def plot_hydraulic_head_gradient(h, kappa):
    U,V = hydraulic_head_gradient(n,h, kappa)

    Y,X = np.mgrid[0:60:60j, 0:60:60j]

    speed = np.sqrt(U**2 + V**2)

    lw = 5*speed / speed.max()
    lw = lw.reshape(n,n)

    plt.imshow(h.reshape(n,n), cmap = 'hot_r')
    plt.colorbar()
    plt.streamplot(X,Y,U.reshape(n,n), V.reshape(n,n), color = 'black', linewidth = lw, density = [0.5,1])
    plt.xticks(np.linspace(0,n-1,7), labels = range(7))
    plt.yticks(np.linspace(0,n-1,7), labels = range(7))
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.xlim(0,n-1)
    plt.ylim(0,n-1)
    plt.title('Steady-state hydraulic head distribution')
    plt.show()


def plot_loop(m):
    ## loop to create training data

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    seed_val = range(m)

    f = source_function(n)


    for seed in seed_val:

        kappa = hydraulic_conductivity_field(n, seed)

        h = solve_darcy_flow(n, kappa, f)

        U,V = hydraulic_head_gradient(n,h, kappa)

        Y,X = np.mgrid[0:60:60j, 0:60:60j]

        speed = np.sqrt(U**2 + V**2)

        lw = 5*speed / speed.max()
        lw = lw.reshape(n,n)

        print('Seed value:', seed)

        fig, (ax1, ax2) = plt.subplots(1,2)

        im = ax1.imshow(np.log(kappa).reshape(n,n), cmap = 'jet', origin = 'lower')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax = cax)
        ax1.set_xticks(np.linspace(0,n-1,7), labels = range(7))
        ax1.set_yticks(np.linspace(0,n-1,7), labels = range(7))
        ax1.set_xlabel('x (km)')
        ax1.set_ylabel('y (km)')
        ax1.set_title('Log hydraulic conductivity')

        im2 = ax2.imshow(h.reshape(n,n), cmap = 'hot_r')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2,cax = cax)
        ax2.streamplot(X,Y,U.reshape(n,n), V.reshape(n,n), color = 'black', linewidth = lw, density = [0.5,1])
        ax2.set_xticks(np.linspace(0,n-1,7), labels = range(7))
        ax2.set_yticks(np.linspace(0,n-1,7), labels = range(7))
        ax2.set_xlabel('x (km)')
        ax2.set_xlim(0,n-1)
        ax2.set_ylim(0,n-1)
        ax2.set_title('Hydraulic head')
        plt.show()



def hydraulic_conductivity_Carrera(n):  # n = 36 in Carrera et al. (1998) paper

    " Generate hydraulic conductivity field k on a n x n grid using Carrera et al. (1998) benchmark. "

    k = np.zeros(n ** 2)

    L = 6
    dx = L / (n - 1)

    def cond(x1, x2):
        if 0 <= x2 < 2 or (2 <= x2 <= 4 and 0 <= x1 < 2):
            return 150
        elif ((2 <= x1 <= 4) and 2 <= x2 <= 4) or (0 <= x1 < 2 and 4 < x2 <= 6):
            return 50
        elif ((2 <= x1 <= 4) and (4 < x2 <= 6)) or ((4 < x1 <= 6) and (2 <= x2 <= 4)):
            return 15
        elif (4 < x1 <= 6) and (4 < x2 <= 6):
            return 5

    k = np.zeros(n ** 2)

    for i in range(n):
        for j in range(n):
            idx = j * n + i
            x1 = i * dx
            x2 = j * dx
            k[idx] = cond(x1, x2)

    return k


def test_hydraulic_conductivity_Carrera(n_carrera = 36):
    kappa_carrera = hydraulic_conductivity_Carrera(n_carrera)

    plt.imshow(kappa_carrera.reshape(n_carrera, n_carrera), cmap='jet', origin='lower')
    plt.title('Hydraulic conductivity field')
    plt.xticks(np.linspace(0, n_carrera - 1, 7), labels=range(7))
    plt.yticks(np.linspace(0, n_carrera - 1, 7), labels=range(7))
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.colorbar()
    plt.show()

    f_carrera = source_function(n_carrera)

    h_carrera = solve_darcy_flow(n_carrera, kappa_carrera, f_carrera)

    fig, ax = plt.subplots()

    CS = ax.contour(h_carrera.reshape(n_carrera, n_carrera), levels=range(100, 200, 5), colors='k')
    ax.clabel(CS, fontsize=9)
    ax.set_xlabel('(a) STEADY STATE')
    plt.title('Steady-state hydraulic head distribution')
    plt.show()


if __name__ == "__main__":
    plot_hydraulic_conductivity_field(60, 9)
    plot_source_function(60, 9)
    test_solve_darcy_flow(60, 9)
    kappa = hydraulic_conductivity_field(60, 9)
    f = source_function(60)
    h = solve_darcy_flow(60, kappa, f)
    plot_hydraulic_head_gradient(h, kappa)
    plot_loop(10)
    test_hydraulic_conductivity_Carrera()
    #fyugyi