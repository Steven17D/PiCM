"""
We refer to the superparticle as particle
"""
from itertools import product

import numpy as np
import scipy.constants as const
from matplotlib import pyplot as plt

from acceptance_rejection import get_random_value
from scipy.stats import truncnorm, maxwell, norm


# k_b = const.physical_constants['Boltzmann constant'][0]
# omega_p = np.sqrt(n_alpha * (q_alpha ** 2) / (m_alpha))
# v_th = np.array([1, 1]) * np.sqrt(k_b * T_alpha * m_alpha)  # Thermal velocity
debye_length = 1  # v_th / omega_p
<<<<<<< HEAD
L = np.array([3, 3]) * debye_length  # Lx and Ly are the lengths of the system in the x and y directions in units of the Debye length.
n = np.array([3, 3])
=======
L = np.array([64, 64]) * debye_length  # Lx and Ly are the lengths of the system in the x and y directions in units of the Debye length.
n = np.array([64, 64])
>>>>>>> bf5b899665addb7b819ae02a0a648732e875a52e
Lx, Ly = L  # size of the system
Nx, Ny = n  # number of grid points
dx, dy = Lx / Nx, Ly / Ny  # delta_x and delta_y
delta_r = L / n  # Vector of delta x and delta y
dxdy = np.multiply.reduce(delta_r)
q = 1  # Charge of a cell? TODO: Check
ro_c = q / dxdy
assert 0.5 * delta_r[0] < debye_length
assert 0.5 * delta_r[1] < debye_length

<<<<<<< HEAD
N = 3
=======
N = 256
>>>>>>> bf5b899665addb7b819ae02a0a648732e875a52e
v_th = 1
v_d = 5 * v_th
n_e = 0.5  # TODO: Check N / np.multiply.reduce(L)
B = np.zeros(3)
q_over_m = 1


def maxwell_distribution(v):
    return (
        (n_e / np.sqrt(2 * np.pi * (v_th ** 2))) * (
            np.exp((-(v - v_d) ** 2)/(2 * (v_th ** 2))) + np.exp((-(v + v_d) ** 2)/(2 * (v_th ** 2)))
        )
    )


def fft(v: np.ndarray) -> np.ndarray:
    return np.fft.fft(v)


def ifft(v: np.ndarray) -> np.ndarray:
    return np.fft.ifft(v)


def density(positions: np.ndarray):
    """
    Calculate the density of particles
    TODO: add charges list
    :param positions:
    :return:
    """
<<<<<<< HEAD
    rho = np.empty(n)
=======
    rho = np.array(n)
>>>>>>> bf5b899665addb7b819ae02a0a648732e875a52e
    for p in range(N):
        i, j = np.floor(positions[p] / delta_r).astype(int)
        h = positions[p] - np.array([i, j]) * delta_r
        h_n = delta_r - h
        rho[i, j] = ro_c * (np.multiply.reduce(h_n) / dxdy)
<<<<<<< HEAD
        a = rho[i, j]
        rho[i, (j+1) % n[1]] = ro_c * (h_n[0] * h[1] / dxdy)
        rho[(i+1) % n[0], j] = ro_c * (h_n[1] * h[0] / dxdy)
        rho[(i+1) % n[0], (j+1) % n[1]] = ro_c * (np.multiply.reduce(h) / dxdy)

=======
        rho[i, (j+1) % n[1]] = ro_c * (h_n[0] * h[1] / dxdy)
        rho[(i+1) % n[0], j] = ro_c * (h_n[1] * h[0] / dxdy)
        rho[(i+1) % n[0], (j+1) % n[1]] = ro_c * (np.multiply.reduce(h) / dxdy)
>>>>>>> bf5b899665addb7b819ae02a0a648732e875a52e
    return rho


def potential(rho: np.ndarray):
    """
    Calculate the potential (phi) from the charge density (rho)
    TODO: Write the proof of the FFT from rho to phi
    :param rho:
    :return:
    """
    rho = rho.astype(complex)

    # FFT rho to rho_k
    rho_k = np.empty_like(rho, dtype=complex)
    for xi in range(n[0]):
        rho_k[xi, :] = fft(rho[xi, :])
    for yi in range(n[1]):
        rho_k[:, yi] = fft(rho[:, yi])

    # Calculate phi_k from rho_k
    Wx = np.exp(2j * np.pi / n[0])
    Wy = np.exp(2j * np.pi / n[1])
    Wn, Wm = 1, 1
    dx_2, dy_2 = delta_r ** 2

    phi_k = np.empty_like(rho_k, dtype=complex)
    for ni in range(n[0]):
        for m in range(n[1]):
            denom = dy_2 * (2.0 - Wn - 1.0 / Wn) + dx_2 * (2.0 - Wm - 1.0 / Wm)
            if denom:
<<<<<<< HEAD
                phi_k[ni, m] = rho_k[ni, m] * (dx_2 * dy_2) / denom
=======
                phi_k[n, m] = rho_k[n, m] * (dx_2 * dy_2) / denom
>>>>>>> bf5b899665addb7b819ae02a0a648732e875a52e
            Wm *= Wy
        Wn *= Wx

    # Inverse FFT phi_k to phi
    phi = np.empty_like(phi_k, dtype=complex)
    for xi in range(n[0]):
        phi[xi, :] = ifft(phi_k[xi, :])
    for yi in range(n[1]):
        phi[:, yi] = ifft(phi_k[:, yi])

    return np.real(phi)


def field_nodes(phi: np.ndarray):
    E = np.zeros([*phi.shape, 3])

    for j in range(n[1]):
        for i in range(n[0]):
            nxt_i = (i + 1) % n[0]
            prv_i = (i - 1) % n[0]
            E[i][j][0] = (phi[prv_i][j] - phi[nxt_i][j]) / (delta_r[0] * 2.0)

    for i in range(n[0]):
        for j in range(n[1]):
            nxt_j = (j + 1) % n[1]
            prv_j = (j - 1) % n[1]
            E[i][j][1] = (phi[i][prv_j] - phi[i][nxt_j]) / (delta_r[1] * 2.0)

    return E


def field_particles(field: np.ndarray, positions: np.array):
    """
    TODO: Add moves
    :param field:
    :param positions:
    :return:
    """
    dx, dy = delta_r
    Nx, Ny = n
<<<<<<< HEAD
    E = np.empty([N, 3])
    for p in range(N):
        i = int(np.floor(positions[p][0] / dx))
        j = int(np.floor(positions[p][1] / dy))
        hx = positions[p][0] - (i * dx)
        hy = positions[p][1] - (j * dy)
        nxt_i = int((i + 1) % Nx)
        nxt_j = int((j + 1) % Ny)
=======
    E = np.empty_like([N, 3])
    for p in range(N):
        i = np.floor(positions[p][0] / dx)
        j = np.floor(positions[p][1] / dy)
        hx = positions[p][0] - (i * dx)
        hy = positions[p][1] - (j * dy)
        nxt_i = (i + 1) % Nx
        nxt_j = (j + 1) % Ny
>>>>>>> bf5b899665addb7b819ae02a0a648732e875a52e
        A = (dx - hx) * (dy - hy)
        B = (dx - hx) * hy
        C = hx * (dy - hy)
        D = hx * hy
        E[p][0] = field[i][j][0] * A + field[i][nxt_j][0] * B + field[nxt_i][j][0] * C + field[nxt_i][nxt_j][0] * D
        E[p][1] = field[i][j][1] * A + field[i][nxt_j][1] * B + field[nxt_i][j][1] * C + field[nxt_i][nxt_j][1] * D

    return E / (dx * dy)


def boris(velocities, E, dt, direction):
    dt = 0.5 * direction * dt
    for p in range(N):
        t = 0.5 * q_over_m * B * dt
        t_2 = np.linalg.norm(t) * np.linalg.norm(t)
        s = (2.0 * t) / (1.0 + t_2)
        v_minus = velocities[p] + 0.5 * q_over_m * E[p] * dt
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        velocities[p] = v_plus + 0.5 * q_over_m * E[p] * dt
    return velocities


def update(positions, velocities, E, dt):
    velocities = boris(velocities, E, dt, direction=1)
    for p in range(N):
<<<<<<< HEAD
        positions[p] += velocities[p][:2] * dt
=======
        positions[p] += velocities[p] * dt
>>>>>>> bf5b899665addb7b819ae02a0a648732e875a52e
        positions[p][0] = positions[p][0] % L[0]
        positions[p][1] = positions[p][1] % L[1]
    return velocities, positions


def setup():
    positions = np.random.uniform(0, Lx, [N, 2])
    velocities = np.zeros([N, 3])
    velocities[:, 0] = [get_random_value(maxwell_distribution, -v_d*2, v_d*2, v_d) for _ in range(N)]
    return positions, velocities

"""
TODO:
Change particle for to:
A = [x1, x2, x3, ...]
A1 = round(A/dx)  # This is the affected cell
A2 = (A1 + 1) % Lx  # This is right
A3 = (A1 - 1) % Lx  # This is left
"""
def main():
    positions, velocities = setup()
    steps = 1000
    dt = 0.1

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(positions[:, 0], positions[:, 1])
    ax[1].scatter(positions[:, 0], velocities[:, 0])
    plt.show()

    for step in range(steps):
        rho = density(positions)
        phi = potential(rho)
        e_field_n = field_nodes(phi)
        e_field_p = field_particles(e_field_n, positions)
        if step == 0:
            velocities = boris(velocities, e_field_p, dt, direction=-1)
        velocities, positions = update(positions, velocities, e_field_p, dt)
        velocities = boris(velocities, e_field_p, dt, direction=1)


if __name__ == '__main__':
    main()
