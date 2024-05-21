"""
We refer to the superparticle as particle
"""
import time

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from PiCM.loader import local_initial_state
from PiCM.simulation import simulate, density, potential
from PiCM.acceptance_rejection import get_random_value

matplotlib.style.use('classic')


def maxwell_distribution(v, v_d, v_th):
    """
    Velocity distribution
    """
    n_e = 0.5  # In order to normalize area
    return (
            (n_e / np.sqrt(2 * np.pi * (v_th ** 2))) * (
            np.exp((-(v - v_d) ** 2) / (2 * (v_th ** 2))) + np.exp((-(v + v_d) ** 2) / (2 * (v_th ** 2)))
    )
    )


def setup(L, v_d, v_th, N):
    positions = np.array([np.random.uniform(0, l, [N]) for l in L]).T
    velocities = np.zeros([N, 3])
    vel_zero = np.zeros(int(N / 2))
    vel = [get_random_value(lambda v: maxwell_distribution(v, v_d, v_th), -v_d * 2, v_d * 2, maxwell_distribution(v_d, v_d, v_th)) for _ in range(N // 2)]
    velocities[:, 0] = np.concatenate((vel_zero, vel))
    q_m = np.concatenate((np.ones(int(N / 2)), -np.ones(int(N / 2))))
    moves = np.concatenate((np.zeros(int(N / 2)), np.ones(int(N / 2))))
    charges = (L[0] * L[1] * q_m) / N
    masses = charges / q_m
    return positions, velocities, q_m, charges, masses, moves


def setup_from_file(L):
    positions, velocities, q_m, moves = local_initial_state(r"tests/electrosctatic/two_stream.dat")
    N = 99856
    charges = (L[0] * L[1] * q_m) / N
    masses = charges / q_m
    return positions, velocities, q_m, charges, masses, moves


def calculate_kinetic_energy(velocities, masses):
    return (masses * (velocities[:, 0] ** 2 + velocities[:, 1] ** 2)).sum() / 2


def main():
    # Table II.
    debye_length = 1
    v_th = 1
    L = np.array([1, 1]) * 64 * debye_length  # size of the system
    n = np.array([1, 1]) * 64
    dt = 0.1
    steps = 500
    v_d = 5.0 * v_th  # Drift velocity
    N = 100000

    delta_r = L / n  # Vector of delta x and delta y
    assert 0.5 * delta_r[0] < debye_length
    assert 0.5 * delta_r[1] < debye_length

    B = np.array([0, 0, 0])

    positions, velocities, q_m, charges, masses, moves = setup(L, v_d, v_th, N)
    movers = moves == 1
    moving_masses = masses[movers]
    color = np.where(velocities[movers, 0] < 0, 'b', 'r')
    Nd = 10

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    ax_vx = ax[0, 0]
    ax_phi = ax[0, 1]
    ax_xy = ax[0, 2]
    ax_vx_h = ax[1, 0]
    ax_energy = ax[1, 1]

    ax_vx.set_xlim([0, L[0]])
    ax_vx.set_ylim([-v_d * 3, v_d * 3])
    ax_vx.set_xlabel(r"$x / \lambda_D$")
    ax_vx.set_ylabel(r"$v_x / v_{th}$")
    ax_vx.grid()

    Np = 40  # Amount of particles
    xy_scatter = ax_xy.scatter(positions[movers, 0][::Np], positions[movers, 1][::Np], c=color[::Np], s=5, linewidth=0)
    ax_xy.set_xlim([0, L[0]])
    ax_xy.set_ylim([0, L[1]])
    ax_xy.set_xlabel(r"$x / \lambda_D$")
    ax_xy.set_ylabel(r"$y / \lambda_D$")
    ax_xy.grid()

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

    scatter = ax_vx.scatter(positions[movers, 0][::Nd], velocities[movers, 0][::Nd],
                            c=color[::Nd], s=5, linewidth=0)

    metadata = dict(title=f'PiCM: N={N}, Steps={steps}')
    writer = FFMpegWriter(fps=24, metadata=metadata)
    max_phi = min_phi = 0
    with writer.saving(fig, f'output/{time.strftime("%Y%m%d-%H%M%S")}.mp4', 200):
        for positions, velocities, rho, phi, e_field_n, step in \
                simulate(positions, velocities, q_m, charges, moves, L, n, delta_r, B, dt, steps):
            if step % 1 != 0:
                continue
            fig.suptitle(r"$\omega_{\rm{pe}}$" + f"$t = {(step * dt):.2f}$")
            color_map.update({'array': phi.T.ravel()})
            min_phi, max_phi = min(min_phi, np.min(phi)), max(max_phi, np.max(phi))
            color_map.set_clim(min_phi, max_phi)

            scatter.set_offsets(np.c_[positions[:, 0][::Nd], velocities[:, 0][::Nd]])
            xy_scatter.set_offsets(np.c_[positions[:, 0][::Np], positions[:, 1][::Np]])

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
            if step % 10 == 0:
                plt.show()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
