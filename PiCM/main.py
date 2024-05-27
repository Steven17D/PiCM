"""
We refer to the superparticle as particle
"""
import time

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from PiCM.simulation import simulate, density, potential
from PiCM.probability import get_random_value, maxwell_distribution

matplotlib.style.use('classic')


def setup(L, v_d, v_th, N):
    two_streams = False
    positions = np.array([np.random.uniform(0, l, [N]) for l in L]).T
    velocities = np.zeros([N, 3])
    vel_zero = np.zeros(int(N / 2))
    velx = [get_random_value(lambda v: maxwell_distribution(v, v_d, v_th), -v_d * 2, v_d * 2,
                             maxwell_distribution(v_d, v_d, v_th)) for _ in range(N // 2)]
    velocities[:, 0] = np.concatenate((vel_zero, velx))
    if two_streams:
        vely = [get_random_value(lambda v: maxwell_distribution(v, v_d, v_th), -v_d * 2, v_d * 2,
                                 maxwell_distribution(v_d, v_d, v_th)) for _ in range(N // 2)]
        velocities[:, 1] = np.concatenate((vel_zero, vely))
    q_m = np.concatenate((np.ones(int(N / 2)), -np.ones(int(N / 2))))
    moves = np.concatenate((np.zeros(int(N / 2)), np.ones(int(N / 2))))
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
    steps = 501
    v_d = 5.0 * v_th  # Drift velocity
    N = 100000

    delta_r = L / n  # Vector of delta x and delta y
    assert 0.5 * delta_r[0] < debye_length
    assert 0.5 * delta_r[1] < debye_length

    B = np.array([0, 0, 0])

    positions, velocities, q_m, charges, masses, moves = setup(L, v_d, v_th, N)
    movers = moves == 1
    moving_masses = masses[movers]
    vx_color = np.where(velocities[movers, 0] < 0, 'b', 'r')
    vy_color = np.where(velocities[movers, 1] < 0, 'b', 'r')
    Nd = 10

    font = dict(fontsize=22)
    fig, ax = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)
    ax_vx = ax[0, 0]
    ax_vy = ax[0, 1]
    ax_phi = ax[0, 2]
    ax_xy = ax[0, 3]
    ax_vx_h = ax[1, 0]
    ax_vy_h = ax[1, 1]
    ax_rho = ax[1, 2]
    ax_energy = ax[1, 3]

    ax_vx.set_xlim([0, L[0]])
    ax_vx.set_ylim([-v_d * 3, v_d * 3])
    ax_vx.set_xlabel(r"$x / \lambda_D$", **font)
    ax_vx.set_ylabel(r"$v_x / v_{th}$", **font)
    ax_vx.grid()

    ax_vy.set_xlim([0, L[1]])
    ax_vy.set_ylim([-v_d * 3, v_d * 3])
    ax_vy.set_xlabel(r"$y / \lambda_D$", **font)
    ax_vy.set_ylabel(r"$v_y / v_{th}$", **font)
    ax_vy.grid()

    Np = 40  # Amount of particles
    xy_scatter = ax_xy.scatter(positions[movers, 0][::Np], positions[movers, 1][::Np], c=vx_color[::Np], s=5,
                               linewidth=0)
    ax_xy.set_xlim([0, L[0]])
    ax_xy.set_ylim([0, L[1]])
    ax_xy.set_xlabel(r"$x / \lambda_D$", **font)
    ax_xy.set_ylabel(r"$y / \lambda_D$", **font)
    ax_xy.grid()

    ax_rho.set_xlim(0, (n - 1)[0])
    ax_rho.set_ylim(0, (n - 1)[1])
    ax_rho.set_xlabel(r"$x / \lambda_D$", **font)
    ax_rho.set_ylabel(r"$y / \lambda_D$", **font)
    rho = density(positions, charges, n, delta_r)
    rho_color_map = ax_rho.pcolormesh(rho, shading="gouraud", cmap="jet")
    rho_bar = plt.colorbar(rho_color_map, ax=ax_rho)
    rho_bar.set_label(r"$\rho / (e\lambda_D^{-2})$", **font)

    ax_phi.set_xlim(0, (n - 1)[0])
    ax_phi.set_ylim(0, (n - 1)[1])
    ax_phi.set_xlabel(r"$x / \lambda_D$", **font)
    ax_phi.set_ylabel(r"$y / \lambda_D$", **font)
    phi = potential(rho, n, delta_r)
    phi_color_map = ax_phi.pcolormesh(phi, shading="gouraud", cmap="jet")
    bar = plt.colorbar(phi_color_map, ax=ax_phi)
    bar.set_label(r"$\phi / (T_e / e)$", **font)

    times = np.arange(0, steps * dt, dt)
    kinetic_energies = np.empty(steps)
    field_energies = np.empty(steps)
    kinetic_energy_plot, = ax_energy.plot([], [], label="Kinetic")
    field_energy_plot, = ax_energy.plot([], [], label="Field")
    total_total_plot, = ax_energy.plot([], [], label="Total")
    ax_energy.set_xlim([0, steps * dt])
    ax_energy.set_xlabel(r"$\omega_{\rm{pe}}t$", **font)
    ax_energy.set_ylabel(r"$E / (n_0 T_e / \varepsilon_0)$", **font)
    ax_energy.grid()
    ax_energy.legend(loc="best")

    vx_scatter = ax_vx.scatter(positions[movers, 0][::Nd], velocities[movers, 0][::Nd],
                               c=vx_color[::Nd], s=5, linewidth=0)

    vy_scatter = ax_vy.scatter(positions[movers, 1][::Nd], velocities[movers, 1][::Nd],
                               c=vx_color[::Nd], s=5, linewidth=0)

    _, _, vx_bars = ax_vx_h.hist(velocities[:, 0], density=True, range=(-v_d * 3, v_d * 3), bins=50, color="red")
    ax_vx_h.set_ylim([0, 0.22])
    ax_vx_h.set_xlabel(r"$v_x / v_{\rm{th}}$", **font)
    ax_vx_h.grid()

    h, _, vy_bars = ax_vy_h.hist(velocities[:, 1], density=True, range=(-v_d * 3, v_d * 3), bins=50, color="red")
    ax_vy_h.set_ylim([0, 0.22])
    ax_vy_h.set_xlabel(r"$v_y / v_{\rm{th}}$", **font)
    ax_vy_h.grid()

    metadata = dict(title=f'PiCM: N={N}, Steps={steps}')
    writer = FFMpegWriter(fps=24, metadata=metadata)
    max_rho = min_rho = 0
    max_phi = min_phi = 0
    name = f'output/{time.strftime("%Y%m%d-%H%M%S")}'
    with writer.saving(fig, f'{name}.mp4', 200):
        for positions, velocities, rho, phi, e_field_n, step in \
                simulate(positions, velocities, q_m, charges, moves, L, n, delta_r, B, dt, steps):
            if step % 1 != 0:
                continue
            fig.suptitle(r"$\omega_{\rm{pe}}$" + f"$t = {(step * dt):.2f}$", **font)

            rho_color_map.update({'array': rho.T.ravel()})
            min_rho, max_rho = min(min_rho, np.min(rho)), max(max_rho, np.max(rho))
            rho_color_map.set_clim(min_rho, max_rho)

            phi_color_map.update({'array': phi.T.ravel()})
            min_phi, max_phi = min(min_phi, np.min(phi)), max(max_phi, np.max(phi))
            phi_color_map.set_clim(min_phi, max_phi)

            vx_scatter.set_offsets(np.c_[positions[:, 0][::Nd], velocities[:, 0][::Nd]])
            vy_scatter.set_offsets(np.c_[positions[:, 1][::Nd], velocities[:, 1][::Nd]])
            xy_scatter.set_offsets(np.c_[positions[:, 0][::Np], positions[:, 1][::Np]])

            n, _ = np.histogram(velocities[:, 0], 50, density=True, range=(-v_d * 3, v_d * 3))
            for count, rect in zip(n, vx_bars.patches):
                rect.set_height(count)
            ax_vx_h.set_ylim([0, max(n.max(), 0.20) * 1.1])

            n, _ = np.histogram(velocities[:, 1], 50, density=True, range=(-v_d * 3, v_d * 3))
            for count, rect in zip(n, vy_bars.patches):
                rect.set_height(count)
            ax_vy_h.set_ylim([0, max(n.max(), 0.20) * 1.1])

            kinetic_energies[step] = calculate_kinetic_energy(velocities, moving_masses)
            field_energies[step] = (rho * phi).sum() * 0.5
            total_energy = kinetic_energies[:step + 1] + field_energies[:step + 1]
            kinetic_energy_plot.set_data(times[:step + 1], kinetic_energies[:step + 1])
            field_energy_plot.set_data(times[:step + 1], field_energies[:step + 1])
            total_total_plot.set_data(times[:step + 1], total_energy)
            ax_energy.set_ylim([0, total_energy.max() * 1.1])

            writer.grab_frame()

            if step % 50 == 0:
                fig.savefig(f"{name}-{step}.png", dpi=500)

            if step == 20:
                plt.show()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
