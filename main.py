"""
We refer to the superparticle as particle
"""
from itertools import product

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, animation
from line_profiler_pycharm import profile
from matplotlib.animation import FFMpegWriter
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
    return (rho * (rho_c / (dxdy ** 2))).T * (n[0] * n[1] / positions.shape[0])


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
    s = (2.0 * u) / (1.0 + np.linalg.norm(u, axis=1) ** 2)[:, np.newaxis]
    qEt2m = 0.5 * charges[:, np.newaxis] * E * dt
    v_minus = velocities + qEt2m
    v_prime = v_minus + np.cross(v_minus, u)
    v_plus = v_minus + np.cross(v_prime, s)
    return (v_plus + qEt2m) * moves[:, np.newaxis]


def update(positions, velocities, charges, moves, E, B, L, dt):
    velocities = boris(velocities, charges, moves, E, B, dt)
    return (positions + (velocities[:, slice(0, 2)] * dt)) % L, velocities


def simulate(positions, velocities, charges, moves, L, n, delta_r, dxdy, B, rho_c, dt, steps):
    for step in tqdm(range(steps)):
        rho = density(positions, charges, n, delta_r, rho_c, dxdy)
        phi = potential(rho, n, delta_r)
        e_field_n = field_nodes(phi, n, delta_r)
        e_field_p = field_particles(e_field_n, positions, moves, n, delta_r)
        if step == 0:
            velocities = boris(velocities, charges, moves, e_field_p, B, -0.5 * dt)
        positions, velocities = update(positions, velocities, charges, moves, e_field_p, B, L, dt)
        velocities = boris(velocities, charges, moves, e_field_p, B, 0.5 * dt)
        yield positions, velocities, rho, phi, e_field_n, step


def setup(L, v_d, N):
    positions = np.array([np.random.uniform(0, l, [N]) for l in L]).T
    velocities = np.zeros([N, 3])
    vel_zero = np.zeros(int(N / 2))
    vel_left = np.random.normal(-v_d, 1, size=int(N / 4))
    vel_right = np.random.normal(v_d, 1, size=int(N / 4))
    velocities[:, 0] = np.concatenate((vel_zero, vel_left, vel_right))
    charges = np.concatenate((np.ones(int(N / 2)), -np.ones(int(N / 2))))
    moves = np.concatenate((np.zeros(int(N / 2)), np.ones(int(N / 2))))
    return positions, velocities, charges, moves


def main():
    # Table II.
    L = np.array([1, 1]) * 64 * debye_length  # size of the system
    n = np.array([1, 1]) * 64
    dt = 0.1  # TODO: Change 0.05 / omega_pe
    steps = 500
    v_d = 5.0 * v_th  # Drift velocity
    N = 100000

    delta_r = L / n  # Vector of delta x and delta y
    dxdy = np.multiply.reduce(delta_r)
    q = 1  # Charge of a cell? TODO: Check
    rho_c = q / dxdy
    assert 0.5 * delta_r[0] < debye_length
    assert 0.5 * delta_r[1] < debye_length

    B = np.array([0, 0, 0])

    positions, velocities, charges, moves = setup(L, v_d, N)
    movers = np.where(moves == 1)
    color = np.where(velocities[movers, 0] < 0, 'b', 'r')

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_xlim([0, L[0]])
    ax[0, 0].set_ylim([-v_d * 2, v_d * 2])
    ax[0, 0].set_xlabel(r"$x / \lambda_D$")
    ax[0, 0].set_ylabel(r"$v_x / v_{th}$")

    ax[0, 1].set_title(r"$\omega_{\rm{pe}}$")
    ax[0, 1].set_xlim(0, (L - delta_r)[0])
    ax[0, 1].set_ylim(0, (L - delta_r)[1])
    ax[0, 1].set_xlabel(r"$x / \lambda_D$")
    ax[0, 1].set_ylabel(r"$y / \lambda_D$")
    rho = density(positions, charges, n, delta_r, rho_c, dxdy)
    phi = potential(rho, n, delta_r)
    color_map = ax[0, 1].pcolormesh(phi, shading="gouraud", cmap="jet", vmin=-25, vmax=25)
    bar = plt.colorbar(color_map, ax=ax[0, 1])
    bar.set_label(r"$\phi / (T_e / e)$")

    scatter = ax[0, 0].scatter(positions[movers, 0].T.squeeze(), velocities[movers, 0].T.squeeze(),
                               c=color.T.squeeze(), s=5, linewidth=0)

    metadata = dict(title='Movie', artist='codinglikemad')
    writer = FFMpegWriter(fps=24, metadata=metadata)
    max_phi = 0
    min_phi = 0
    with writer.saving(fig, 'Movie.mp4', 200):
        for positions, velocities, rho, phi, e_field_n, step in \
                simulate(positions, velocities, charges, moves, L, n, delta_r, dxdy, B, rho_c, dt, steps):
            if step % 1 != 0:
                continue
            color_map.update({'array': phi.ravel()})
            scatter.set_offsets(np.c_[positions[movers, 0].T.squeeze(), velocities[movers, 0].T.squeeze()])
            fig.canvas.draw_idle()
            writer.grab_frame()
            min_phi = min(min_phi, np.min(phi))
            max_phi = max(max_phi, np.max(phi))
    print('min_phi', min_phi)
    print('max_phi', max_phi)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
