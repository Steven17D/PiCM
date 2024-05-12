"""
We refer to the superparticle as particle
"""
from itertools import product

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, animation
from line_profiler_pycharm import profile
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
            np.exp((-(v - v_d) ** 2) / (2 * (v_th ** 2))) + np.exp((-(v + v_d) ** 2) / (2 * (v_th ** 2)))
    )
    )


def fft(v: np.ndarray) -> np.ndarray:
    return np.fft.fft(v)


def ifft(v: np.ndarray) -> np.ndarray:
    return np.fft.ifft(v)


rhos = None  # Rhos are initialized once for performance


def density(positions: np.ndarray, charges, n, delta_r, rho_c, dxdy):
    """
    Calculate the density of particles
    sub_rho = np.rollaxis(np.array(sub_rho), 1)
    sub_rho = sub_rho.reshape((positions.shape[0], 2, 2))
    sub_rho_padded = np.pad(sub_rho, ((0, 0), (0, n[0]-2), (0, n[1]-2)), 'constant', constant_values=(0,))
    rhos = np.roll(np.roll(sub_rho_padded, ijs[:, 0], axis=1), ijs[:, 1], axis=2)

    :param positions:
    :return:
    """
    global rhos
    rho = np.zeros(n)
    ijs = np.floor(positions / delta_r).astype(int)  # TODO: Check round of floor
    ijs_up = (ijs + [0, 1]) % n
    ijs_right = (ijs + [1, 0]) % n
    ijs_diag = (ijs + [1, 1]) % n
    h = positions - ijs * delta_r
    h_n = delta_r - h
    if rhos is None:
        rhos = np.full((positions.shape[0], *rho.shape), rho)

    origin_values = h_n[:, 0] * h_n[:, 1] * charges
    up_values = h_n[:, 0] * h[:, 1] * charges
    right_values = h[:, 0] * h_n[:, 1] * charges
    diag_values = h[:, 0] * h[:, 1] * charges

    sub_rho = np.concatenate([origin_values, up_values, right_values, diag_values])
    indices = np.array([ijs, ijs_up, ijs_right, ijs_diag])

    rho_index = np.arange(positions.shape[0])
    rho_index = np.broadcast_to(rho_index, (4, *rho_index.shape))
    flat_indices = indices[:, :, 0] + rho.shape[1] * indices[:, :, 1] + rho.size * rho_index

    np.put(rhos, flat_indices, sub_rho)
    np.add.reduce(rhos, axis=0, out=rho)
    np.put(rhos, flat_indices, 0)

    return (rho * (rho_c / (dxdy ** 2))).T / 24.37889


def potential(rho: np.ndarray, n, delta_r):
    """
    Calculate the potential (phi) from the charge density (rho)
    TODO: Write the proof of the FFT from rho to phi
    :param rho:
    :return:
    """
    rho = rho.astype(complex)

    # FFT rho to rho_k
    for xi in range(n[0]):
        rho[xi, :] = fft(rho[xi, :])
    for yi in range(n[1]):
        rho[:, yi] = fft(rho[:, yi])

    rho_k = rho
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
    for xi in range(n[0]):
        phi_k[xi, :] = ifft(phi_k[xi, :])
    for yi in range(n[1]):
        phi_k[:, yi] = ifft(phi_k[:, yi])

    return np.real(phi_k) + 0.0014421573082545325


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


def field_particles(field: np.ndarray, positions: np.array, moves, n, delta_r):
    """
    TODO: Add moves
    :param field:
    :param positions:
    :return:
    """
    dx, dy = delta_r
    ijs = np.floor(positions / dy).astype(int)
    h = positions - ijs * delta_r
    nxt_ijs = (ijs + 1) % n
    A = ((dx - h[:, 0]) * (dy - h[:, 1]))[:, np.newaxis] * field[ijs[:, 0], ijs[:, 1]]
    B = ((dx - h[:, 0]) * h[:, 1])[:, np.newaxis] * field[ijs[:, 0], nxt_ijs[:, 1]]
    C = (h[:, 0] * (dy - h[:, 1]))[:, np.newaxis] * field[nxt_ijs[:, 0], ijs[:, 1]]
    D = (h[:, 0] * h[:, 1])[:, np.newaxis] * field[nxt_ijs[:, 0], nxt_ijs[:, 1]]
    E = A + B + C + D
    E = E * moves[:, np.newaxis]
    return E / (dx * dy)


def boris(velocities, charges, moves, E, B, dt):
    u = 0.5 * charges[:, np.newaxis] * B * dt
    s = (2.0 * u) / (1.0 + np.linalg.norm(u, axis=1)**2)[:, np.newaxis]
    qEt2m = 0.5 * charges[:, np.newaxis] * E * dt
    v_minus = velocities + qEt2m
    v_prime = v_minus + np.cross(v_minus, u)
    v_plus = v_minus + np.cross(v_prime, s)
    return (v_plus + qEt2m) * moves[:, np.newaxis]


def update(positions, velocities, charges, moves, E, B, L, dt):
    velocities = boris(velocities, charges, moves, E, B, dt)
    return (positions + (velocities[:, slice(0, 2)] * dt)) % L, velocities


def setup(Lx, Ly, v_d, N):
    positions = np.array([np.random.uniform(0, l, [N]) for l in [Lx, Ly]]).T
    velocities = np.zeros([N, 3])
    # velocities[:, 0] = list(sorted([get_random_value(maxwell_distribution, -v_d * 2, v_d * 2, v_d) for _ in range(N)]))
    vel_zero = np.zeros(int(N / 2))
    vel_left = np.random.normal(-v_d, 1, size=int(N / 4))
    vel_right = np.random.normal(v_d, 1, size=int(N / 4))
    velocities[:, 0] = np.concatenate((vel_zero, vel_left, vel_right))
    charges = np.concatenate((np.ones(int(N / 2)), -np.ones(int(N / 2))))
    moves = np.concatenate((np.zeros(int(N / 2)), np.ones(int(N / 2))))
    return positions, velocities, charges, moves


def main():
    # Table II.
    Lx = Ly = 64 * debye_length  # size of the system
    n_x = n_y = 64
    dx, dy = Lx / n_x, Ly / n_y  # delta_x and delta_y
    dt = 0.1  # TODO: Change 0.05 / omega_pe
    steps = 500
    v_d = 5.0 * v_th  # Drift velocity
    N = 100000  # TODO: Change to 10 ** 6

    L = np.array([Lx, Ly])
    n = np.array([n_x, n_y])
    delta_r = L / n  # Vector of delta x and delta y
    dxdy = np.multiply.reduce(delta_r)
    q = 1  # Charge of a cell? TODO: Check
    rho_c = q / dxdy
    assert 0.5 * delta_r[0] < debye_length
    assert 0.5 * delta_r[1] < debye_length

    B = np.zeros(3)
    B[2] = 0.1

    fig, ax = plt.subplots()

    positions, velocities, charges, moves = setup(Lx, Ly, v_d, N)
    # plot_positions_and_velocities(ax, Lx, Ly, positions, v_d, velocities, charges, 0)
    artists = []
    for step in tqdm(range(steps)):
        rho = density(positions, charges, n, delta_r, rho_c, dxdy)
        phi = potential(rho, n, delta_r)
        e_field_n = field_nodes(phi, n, delta_r)
        e_field_p = field_particles(e_field_n, positions, moves, n, delta_r)
        if step == 0:
            velocities = boris(velocities, charges, moves, e_field_p, B, -0.5 * dt)
        positions, velocities = update(positions, velocities, charges, moves, e_field_p, B, L, dt)
        velocities = boris(velocities, charges, moves, e_field_p, B, 0.5 * dt)
        if step % 1 == 0:
            artists.append(plot_color_mesh(ax, Lx, Ly, dx, dy, phi))

    # plot_positions_and_velocities(ax, Lx, Ly, positions, v_d, velocities, charges, 1)
    # ax.set_title(r"$\omega_{\rm{pe}}$", fontsize=25)
    # bar = plt.colorbar(color_map, ax=plt.gca())
    ax.set_xlim(0, Lx - dx)
    ax.set_ylim(0, Ly - dy)
    ax.set_xlabel(r"$x / \lambda_D$", fontsize=25)
    ax.set_ylabel(r"$y / \lambda_D$", fontsize=25)
    # bar.set_label(r"$\phi / (T_e / e)$", fontsize=25)
    artists.append(plot_color_mesh(ax, Lx, Ly, dx, dy, phi))
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    ani.save('test.mp4', fps=24)
    plt.close()


def plot_color_mesh(ax, Lx, Ly, dx, dy, phi):
    x, y = np.meshgrid(np.arange(0, Lx), np.arange(0, Ly))
    color_map = ax.pcolormesh(x, y, phi, shading="gouraud", cmap="jet")

    # ax.gca().set_aspect("equal")
    return [color_map]

def plot_positions_and_velocities(ax, Lx, Ly, positions, v_d, velocities, charges, i):
    ax[0, i].scatter(positions[:, 0], positions[:, 1], c=np.where(charges < 0, 'b', 'r'), s=5, linewidth=0)
    ax[0, i].set_xlim([0, Lx])
    ax[0, i].set_ylim([0, Ly])
    ax[0, i].set_xlabel(r"$x / \lambda_D$")
    ax[0, i].set_ylabel(r"$y / \lambda_D$")
    ax[1, i].scatter(positions[:, 0], velocities[:, 0], c=np.where(charges < 0, 'b', 'r'), s=5, linewidth=0)
    ax[1, i].set_xlim([0, Lx])
    ax[1, i].set_ylim([-v_d * 2, v_d * 2])
    ax[1, i].set_xlabel(r"$x / \lambda_D$")
    ax[1, i].set_ylabel(r"$v_x / v_{th}$")


if __name__ == '__main__':
    main()
