"""
We refer to the superparticle as particle
"""
import time
from itertools import product

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, animation
from line_profiler_pycharm import profile
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from loader import local_initial_state

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


def density(positions: np.ndarray, charges, n, delta_r):
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
    return (rho / (delta_r[0] * delta_r[0] * delta_r[1] * delta_r[1])).T


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


def field_particles(field: np.ndarray, positions: np.array, n, delta_r):
    dx, dy = delta_r
    ijs = np.floor(positions / dy).astype(int)
    h = positions - ijs * delta_r
    nxt_ijs = (ijs + 1) % n
    A = ((dx - h[:, 0]) * (dy - h[:, 1]))[:, np.newaxis] * field[ijs[:, 0], ijs[:, 1]]
    B = ((dx - h[:, 0]) * h[:, 1])[:, np.newaxis] * field[ijs[:, 0], nxt_ijs[:, 1]]
    C = (h[:, 0] * (dy - h[:, 1]))[:, np.newaxis] * field[nxt_ijs[:, 0], ijs[:, 1]]
    D = (h[:, 0] * h[:, 1])[:, np.newaxis] * field[nxt_ijs[:, 0], nxt_ijs[:, 1]]
    E = A + B + C + D
    return E / (dx * dy)


def boris(velocities, q_m, E, B, dt):
    u = 0.5 * q_m[:, np.newaxis] * B * dt
    s = (2.0 * u) / (1.0 + np.linalg.norm(u, axis=1) ** 2)[:, np.newaxis]
    qEt2m = 0.5 * q_m[:, np.newaxis] * E * dt
    v_minus = velocities + qEt2m
    v_prime = v_minus + np.cross(v_minus, u)
    v_plus = v_minus + np.cross(v_prime, s)
    return v_plus + qEt2m


def update(positions, velocities, q_m, E, B, L, dt):
    velocities = boris(velocities, q_m, E, B, dt)
    return (positions + (velocities[:, slice(0, 2)] * dt)) % L, velocities


def simulate(positions, velocities, q_m, charges, moves, L, n, delta_r, B, dt, steps):
    statics = moves == 0
    static_rho = density(positions[statics], charges[statics], n, delta_r)
    moving = moves == 1
    moving_positions = positions[moving]
    moving_charges = charges[moving]
    moving_velocities = velocities[moving]
    moving_q_m = q_m[moving]
    for step in tqdm(range(steps)):
        rho = static_rho + density(moving_positions, moving_charges, n, delta_r)
        phi = potential(rho, n, delta_r)
        e_field_n = field_nodes(phi, n, delta_r)
        e_field_p = field_particles(e_field_n, moving_positions, n, delta_r)
        if step == 0:
            moving_velocities = boris(moving_velocities, moving_q_m, e_field_p, B, -0.5 * dt)
        moving_positions, moving_velocities = update(moving_positions, moving_velocities, moving_q_m, e_field_p, B,
                                                     L, dt)
        new_velocities = boris(moving_velocities, moving_q_m, e_field_p, B, 0.5 * dt)
        yield moving_positions, new_velocities, rho, phi, e_field_n, step


def setup(L, v_d, N):
    positions = np.array([np.random.uniform(0, l, [N]) for l in L]).T
    velocities = np.zeros([N, 3])
    vel_zero = np.zeros(int(N / 2))
    vel = [get_random_value(maxwell_distribution, -v_d * 3, v_d * 3, v_d) for _ in range(N // 2)]
    velocities[:, 0] = np.concatenate((vel_zero, vel))
    q_m = np.concatenate((np.ones(int(N / 2)), -np.ones(int(N / 2))))
    moves = np.concatenate((np.zeros(int(N / 2)), np.ones(int(N / 2))))
    charges = (L[0] * L[1] * q_m) / N
    masses = charges / q_m
    return positions, velocities, q_m, charges, masses, moves


def calculate_kinetic_energy(velocities, masses):
    return (masses * (velocities[:, 0] ** 2 + velocities[:, 1] ** 2)).sum() / 2


def main():
    # Table II.
    L = np.array([1, 1]) * 64 * debye_length  # size of the system
    n = np.array([1, 1]) * 64
    dt = 0.1
    steps = 4000
    v_d = 5.0 * v_th  # Drift velocity
    N = 100000

    delta_r = L / n  # Vector of delta x and delta y
    assert 0.5 * delta_r[0] < debye_length
    assert 0.5 * delta_r[1] < debye_length

    B = np.array([0, 0, 0])

    positions, velocities, q_m, charges, masses, moves = setup(L, v_d, N)
    movers = moves == 1
    moving_masses = masses[movers]
    color = np.where(velocities[movers, 0] < 0, 'b', 'r')

    fig, ax = plt.subplots(2, 2)
    ax_vx = ax[0, 0]
    ax_phi = ax[0, 1]
    ax_vx_h = ax[1, 0]
    ax_energy = ax[1, 1]

    ax_vx.set_xlim([0, L[0]])
    ax_vx.set_ylim([-v_d * 3, v_d * 3])
    ax_vx.set_xlabel(r"$x / \lambda_D$")
    ax_vx.set_ylabel(r"$v_x / v_{th}$")
    ax_vx.grid()

    ax_phi.set_title(r"$\omega_{\rm{pe}}$")
    ax_phi.set_xlim(0, (L - delta_r)[0])
    ax_phi.set_ylim(0, (L - delta_r)[1])
    ax_phi.set_xlabel(r"$x / \lambda_D$")
    ax_phi.set_ylabel(r"$y / \lambda_D$")
    rho = density(positions, charges, n, delta_r)
    phi = potential(rho, n, delta_r)
    color_map = ax_phi.pcolormesh(phi, shading="gouraud", cmap="jet", vmin=-16, vmax=21)
    bar = plt.colorbar(color_map, ax=ax_phi)
    bar.set_label(r"$\phi / (T_e / e)$")

    times = np.arange(0, steps * dt, dt)
    kinetic_energies = np.empty(steps)
    field_energies = np.empty(steps)
    kinetic_energy_plot, = ax_energy.plot([], [], label="Kinetic")
    field_energy_plot, = ax_energy.plot([], [], label="Field")
    total_total_plot, = ax_energy.plot([], [], label="Total")
    ax_energy.set_xlim([0, steps * dt])
    ax_energy.set_xlabel(r"$\omega_{\rm{pe}}t$")
    ax_energy.set_ylabel(r"$E / (n_0 T_e / \varepsilon_0)$")
    ax_energy.grid()
    ax_energy.legend(loc="best")
    fig.tight_layout()

    Nd = 10
    scatter = ax_vx.scatter(positions[movers, 0][::Nd], velocities[movers, 0][::Nd],
                            c=color[::Nd], s=5, linewidth=0)

    metadata = dict(title=f'PiCM: N={N}, Steps={steps}')
    writer = FFMpegWriter(fps=24, metadata=metadata)
    max_phi = 0
    min_phi = 0
    with writer.saving(fig, f'{time.strftime("%Y%m%d-%H%M%S")}.mp4', 200):
        for positions, velocities, rho, phi, e_field_n, step in \
                simulate(positions, velocities, q_m, charges, moves, L, n, delta_r, B, dt, steps):
            if step % 1 != 0:
                continue
            fig.suptitle(r"$\omega_{\rm{pe}}$" + f"$t = {(step * dt):.2f}$")
            color_map.update({'array': phi.T.ravel()})
            min_phi, max_phi = min(min_phi, np.min(phi)), max(max_phi, np.max(phi))
            color_map.set_clim(min_phi, max_phi)

            scatter.set_offsets(np.c_[positions[:, 0][::Nd], velocities[:, 0][::Nd]])

            ax_vx_h.cla()
            ax_vx_h.hist(velocities[:, 0], density=True, range=(-v_d * 3, v_d * 3), bins=50,
                         color="red")
            ax_vx_h.set_ylim([0, 0.22])
            ax_vx_h.set_xlabel(r"$v_x / v_{\rm{th}}$")
            ax_vx_h.grid()

            kinetic_energies[step] = calculate_kinetic_energy(velocities, moving_masses)
            field_energies[step] = (rho * phi).sum() * 0.5
            total_energy = kinetic_energies[:step + 1] + field_energies[:step + 1]
            kinetic_energy_plot.set_data(times[:step + 1], kinetic_energies[:step + 1])
            field_energy_plot.set_data(times[:step + 1], field_energies[:step + 1])
            total_total_plot.set_data(times[:step + 1], total_energy)
            ax_energy.set_ylim([0, total_energy.max() * 1.1])

            fig.canvas.draw_idle()
            writer.grab_frame()
    print('min_phi', min_phi)
    print('max_phi', max_phi)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
