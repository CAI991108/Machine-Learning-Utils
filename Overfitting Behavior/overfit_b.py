from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_legendre_coe(K):
    """
    Input:
        K: the highest order polynomial degree
    Return:
        P: (K+1, K+1), the coefficient matrix, where i-th column corresponds
            to i-th legendre polynomial's coefficients.
    """
    # initialize the first two coefficients
    P = [
        np.array([1] + [0] * K),
        np.array([0, 1] + [0] * (K - 1))
    ]
    for k in range(2, K + 1):
        P_k = np.zeros((K + 1,))
        P_k += (2 * k - 1) / k * np.roll(P[-1], 1) - (k - 1) / k * P[-2]
        P.append(P_k)

    return np.array(P).T


def generate_data(Qg, var, n):
    """
    Generate n data samples with Qg order legendre polynomial f and noise level var
    """
    x = np.random.uniform(-1, 1, (n,))
    epsilon = np.random.normal(0, np.sqrt(var), (n,))

    # get f
    normalize_factor = np.sqrt(np.sum([1 / (2 * q + 1) for q in range(Qg + 1)]))
    a = np.random.normal(size=(Qg + 1,)) / normalize_factor  # scale the variance of f to 1

    # get y
    Phi_x_Qg = np.vstack([np.power(x, i) for i in range(Qg + 1)]).T  # (n, Qg+1)
    P = get_legendre_coe(Qg)  # (Qg+1, Qg+1)
    y = Phi_x_Qg @ (P @ a) + epsilon

    return x, y, a


def calBestFitCoefficients(x, y, K):
    """
    calculate the best degree-K legendre polynomial that fits data (x, y)

    Return:
        w_star: (K+1, ), the best-fit coefficients
    """
    Lk = np.array([np.power(x, i) for i in range(K+1)]).T@get_legendre_coe(K)

    # Calculate the pseudo-inverse of Lk
    pseudo = np.linalg.inv(Lk.T @ Lk) @ Lk.T

    # Calculate the best-fit coefficients
    w_star = pseudo @ y
    return w_star


def calErout(w_star, a):
    """
    Input:
        w_star: (K+1, ), the best-fit coefficients
        a: (Qg+1, ), the true coefficients of the legendre polynomial
    Return:
        Erout: scalar, the out-of-sample error
    """
    K = w_star.shape[0] - 1
    Qg = a.shape[0] - 1
    normalize_factor = np.sqrt(np.sum([1 / (2 * q + 1) for q in range(Qg + 1)]))

    # Calculate the minimum degree
    min_degree = min(K, Qg)

    # Calculate the first term
    first = np.sum(w_star ** 2 * np.array([1 / (2 * k + 1) for k in range(K + 1)]))

    # Calculate the second term
    second = -2 * (np.sum(w_star[:min_degree] * a[:min_degree] *
                          np.array([1 / (2 * i + 1) for i in range(min_degree)])) / normalize_factor)

    # Calculate the third term
    third = (np.sum(a ** 2 * np.array([1 / (2 * q + 1) for q in range(Qg + 1)])) / normalize_factor ** 2)

    # Calculate the out-of-sample error
    Erout = 2 * (first + second + third)
    return Erout


repeat_num = 100
ns = np.arange(20, 120, 5)
vars = np.arange(0, 2, 0.05)

logs = {
    "Erout_10": np.zeros((len(ns), len(vars))),
    "Erout_2": np.zeros((len(ns), len(vars))),
    "overfit_measure": np.zeros((len(ns), len(vars))),
}

Qg =20
i, j = 0, 0
for var, n in product(vars, ns):
    Erin_10, Erin_2 = 0, 0
    Erout_10, Erout_2 = 0, 0
    for _ in range(repeat_num):
        x, y, a = generate_data(Qg, var, n)
        w_star_10 = calBestFitCoefficients(x, y, 10)
        w_star_2 = calBestFitCoefficients(x, y, 2)

        Erout_10 += calErout(w_star_10, a) / repeat_num
        Erout_2 += calErout(w_star_2, a) / repeat_num

    overfit_measure = Erout_10 - Erout_2
    logs["Erout_10"][i, j] = Erout_10
    logs["Erout_2"][i, j] = Erout_2
    logs["overfit_measure"][i, j] = overfit_measure

    i += 1
    if i == len(ns):
        i = 0
        j += 1

# clip for better plot view
for key in logs:
    logs[key] = np.clip(logs[key], -2, 10)

# plot
cmap = plt.colormaps.get_cmap("jet")

fig1, ax2 = plt.subplots(constrained_layout=True)
Qf_mesh, n_mesh = np.meshgrid(vars, ns)
CS = ax2.contourf(n_mesh.T, Qf_mesh.T, logs["overfit_measure"].T, cmap=cmap)
ax2.set_title('Impact of $\\sigma$ and $n$')
ax2.set_xlabel('Number of Data Points $n$')
ax2.set_ylabel('Noise level $\\sigma$')

cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('Overfit Measure')
N = cmap.N
CS.cmap.set_under(cmap(1))
CS.cmap.set_over(cmap(N - 1))

plt.savefig("overfit_sigma_vs_n.pdf")
plt.show()
