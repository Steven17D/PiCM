"""
We refer to the superparticle as particle
"""
from itertools import product

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.style.use('classic')

from acceptance_rejection import get_random_value

debye_length = 1
omega_pe = 1
v_th = 1


def maxwell_distribution(v):
    """
    Velocity distribution
    """
    v_d = 5
    n_e = 0.5  # In order to normalize area
    return (
        (n_e / np.sqrt(2 * np.pi * (v_th ** 2))) * (
            np.exp((-(v - v_d) ** 2)/(2 * (v_th ** 2))) + np.exp((-(v + v_d) ** 2)/(2 * (v_th ** 2)))
        )
    )


def fft(v: np.ndarray) -> np.ndarray:
    return np.fft.fft(v)


def ifft(v: np.ndarray) -> np.ndarray:
    return np.fft.ifft(v)


def density(positions: np.ndarray, n, delta_r, rho_c, dxdy):
    """
    Calculate the density of particles
    TODO: add charges list
    :param positions:
    :return:
    """
    rho = np.zeros(n)
    ijs = np.floor(positions / delta_r).astype(int)
    ijs_right = (ijs + [1, 0]) % n
    ijs_up = (ijs + [0, 1]) % n
    ijs_diag = (ijs + [1, 1]) % n
    h = positions - ijs * delta_r
    h_n = delta_r - h
    rhos = np.full((positions.shape[0], *rho.shape), rho)
    sub_rho = [h_n[:, 0] * h_n[:, 1], h_n[:, 0] * h[:, 1], h_n[:, 1] * h[:, 0], h[:, 0] * h[:, 1]]
    indices = np.array([ijs, ijs_up, ijs_right, ijs_diag])
    rho_index = np.arange(positions.shape[0])
    rho_index = np.broadcast_to(rho_index, (4, *rho_index.shape))
    flat_indices = indices[:, :, 0] + rho.shape[1] * indices[:, :, 1] + rho.size * rho_index

    np.put(rhos, flat_indices, sub_rho)
    rho = rhos.sum(axis=0)
    return rho * rho_c / dxdy


def potential(rho: np.ndarray, n, delta_r):
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
                phi_k[ni, m] = rho_k[ni, m] * (dx_2 * dy_2) / denom
            Wm *= Wy
        Wn *= Wx

    # Inverse FFT phi_k to phi
    phi = np.empty_like(phi_k, dtype=complex)
    for xi in range(n[0]):
        phi[xi, :] = ifft(phi_k[xi, :])
    for yi in range(n[1]):
        phi[:, yi] = ifft(phi_k[:, yi])

    return np.real(phi)


def field_nodes(phi: np.ndarray, n, delta_r):
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


def field_particles(field: np.ndarray, positions: np.array, n, delta_r):
    """
    TODO: Add moves
    :param field:
    :param positions:
    :return:
    """
    dx, dy = delta_r
    Nx, Ny = n
    E = np.zeros([len(positions), 3])
    for p, position in enumerate(positions, start=0):
        i = int(np.floor(positions[p][0] / dx))
        j = int(np.floor(positions[p][1] / dy))
        hx = positions[p][0] - (i * dx)
        hy = positions[p][1] - (j * dy)
        nxt_i = int((i + 1) % Nx)
        nxt_j = int((j + 1) % Ny)
        A = (dx - hx) * (dy - hy)
        B = (dx - hx) * hy
        C = hx * (dy - hy)
        D = hx * hy
        E[p][0] = field[i][j][0] * A + field[i][nxt_j][0] * B + field[nxt_i][j][0] * C + field[nxt_i][nxt_j][0] * D
        E[p][1] = field[i][j][1] * A + field[i][nxt_j][1] * B + field[nxt_i][j][1] * C + field[nxt_i][nxt_j][1] * D

    return E / (dx * dy)


def boris(velocities, E, dt, direction, q_over_m, B):
    dt = 0.5 * direction * dt
    t = 0.5 * q_over_m * B * dt
    t_2 = np.linalg.norm(t) * np.linalg.norm(t)
    s = (2.0 * t) / (1.0 + t_2)
    for p, velocity in enumerate(velocities):
        v_minus = velocity + 0.5 * q_over_m * E[p] * dt
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        velocities[p] = v_plus + 0.5 * q_over_m * E[p] * dt
    # return velocities


def update(positions, velocities, E, dt, L, q_over_m, B):
    boris(velocities, E, dt, 1, q_over_m, B)
    for p, position in enumerate(positions, start=0):
        positions[p] += velocities[p][:2] * dt
        positions[p][0] = positions[p][0] % L[0]
        positions[p][1] = positions[p][1] % L[1]
    # return velocities, positions


def setup(Lx, Ly, v_d, N):
    positions = np.array([np.random.uniform(0, l, [N]) for l in [Lx, Ly]]).T
    velocities = np.zeros([N, 3])
    velocities[:, 0] = [get_random_value(maxwell_distribution, -v_d*2, v_d*2, v_d) for _ in range(N)]
    return positions, velocities


def main():
    # Table II.
    Lx = Ly = 64 * debye_length  # size of the system
    n_x = n_y = 64
    dx, dy = Lx / n_x, Ly / n_y  # delta_x and delta_y
    dt = 0.1  # TODO: Change 0.05 / omega_pe
    steps = 1
    v_d = 5.0 * v_th  # Drift velocity
    N = 64000  # TODO: Change to 10 ** 6

    L = np.array([Lx, Ly])
    n = np.array([n_x, n_y])
    delta_r = L / n  # Vector of delta x and delta y
    dxdy = np.multiply.reduce(delta_r)
    q = 1  # Charge of a cell? TODO: Check
    rho_c = q / dxdy
    assert 0.5 * delta_r[0] < debye_length
    assert 0.5 * delta_r[1] < debye_length

    B = np.zeros(3)
    q_over_m = 1

    fig, ax = plt.subplots(2, 2)

    positions, velocities = setup(Lx, Ly, v_d, N)
    plot_positions_and_velocities(ax, Lx, Ly, positions, v_d, velocities, 0)

    for step in tqdm(range(steps)):
        rho = density(positions, n, delta_r, rho_c, dxdy)
        phi = potential(rho, n, delta_r)
        e_field_n = field_nodes(phi, n, delta_r)
        e_field_p = field_particles(e_field_n, positions, n, delta_r)
        if step == 0:
            boris(velocities, e_field_p, dt, -1, q_over_m, B)
        update(positions, velocities, e_field_p, dt, L, q_over_m, B)
        boris(velocities, e_field_p, dt, 1, q_over_m, B)

    plot_positions_and_velocities(ax, Lx, Ly, positions, v_d, velocities, 1)
    plt.show()

    plot_color_mesh(Lx, Ly, dx, dy, phi)


def plot_color_mesh(Lx, Ly, dx, dy, phi):
    x, y = np.meshgrid(np.arange(0, Lx), np.arange(0, Ly))
    plt.figure()
    plt.title(r"$\omega_{\rm{pe}}$", fontsize=25)
    color_map = plt.pcolormesh(x, y, phi, shading="gouraud", cmap="jet")
    bar = plt.colorbar(color_map, ax=plt.gca())
    plt.xlim(0, Lx - dx)
    plt.ylim(0, Ly - dy)
    plt.xlabel(r"$x / \lambda_D$", fontsize=25)
    plt.ylabel(r"$y / \lambda_D$", fontsize=25)
    bar.set_label(r"$\phi / (T_e / e)$", fontsize=25)
    plt.gca().set_aspect("equal")
    plt.show()


def plot_positions_and_velocities(ax, Lx, Ly, positions, v_d, velocities, i):
    ax[0, i].scatter(positions[:, 0], positions[:, 1], c=np.where(velocities[:, 0] < 0, 'b', 'r'), s=5, linewidth=0)
    ax[0, i].set_xlim([0, Lx])
    ax[0, i].set_ylim([0, Ly])
    ax[0, i].set_xlabel(r"$x / \lambda_D$")
    ax[0, i].set_ylabel(r"$y / \lambda_D$")
    ax[1, i].scatter(positions[:, 0], velocities[:, 0], c=np.where(velocities[:, 0] < 0, 'b', 'r'), s=5, linewidth=0)
    ax[1, i].set_xlim([0, Lx])
    ax[1, i].set_ylim([-v_d * 2, v_d * 2])
    ax[1, i].set_xlabel(r"$x / \lambda_D$")
    ax[1, i].set_ylabel(r"$v_x / v_{th}$")


if __name__ == '__main__':
    main()
