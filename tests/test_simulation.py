import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from PiCM.loader import local_initial_state, load_rho, load_field, load_space, load_energy, load_config
from PiCM.simulation import density, boris, field_nodes, field_particles, potential, update, calculate_kinetic_energy


class TestDensity(unittest.TestCase):
    def test_data(self):
        N, L, n = load_config(r"electrosctatic\sim_two_stream.json")
        delta_r = L / n
        positions, _, q_m, _ = local_initial_state(r"electrosctatic\two_stream.dat")
        expected_rho = load_rho(r"electrosctatic\rho\step_0_.dat", n, delta_r)
        charges = (L[0] * L[1] * q_m) / N
        rho = density(positions, charges, n, delta_r)
        np.testing.assert_allclose(rho, expected_rho, rtol=1e-5)


class TestPotential(unittest.TestCase):

    def test_data(self):
        N, L, n = load_config(r"electrosctatic\sim_two_stream.json")
        delta_r = L / n
        rho = load_rho(r"electrosctatic\rho\step_0_.dat", n, delta_r)
        expected_phi = load_rho(r"electrosctatic\phi\step_0_.dat", n, delta_r)
        phi = potential(rho, n, delta_r)
        np.testing.assert_allclose(phi, expected_phi, rtol=0.008)

    def test_many_particles_phi(self):
        N = 10 ** 5
        self._plot_phi(N)

    def _plot_phi(self, N, show_positions=False):
        Lx = Ly = n_x = n_y = 64
        dx = dy = 1
        delta_r = np.array([dx, dy])
        n = np.array([n_x, n_y])
        x, y = np.meshgrid(np.arange(0, Lx), np.arange(0, Ly))
        positions = np.random.uniform(0, Ly, (N, 2))
        charges = np.ones(N)
        rho = density(positions, charges, np.array([Lx, Ly]), delta_r)
        # TODO: Plot rho
        phi = potential(rho, n, delta_r)
        fig, ax = plt.subplots()
        ax.title.set_text(r"$\omega_{\rm{pe}}$")
        color_map = ax.pcolormesh(x, y, phi.T, shading="gouraud", cmap="jet")
        bar = plt.colorbar(color_map)
        ax.set_xlim(0, Lx - dx)
        ax.set_ylim(0, Ly - dy)
        ax.set_xlabel(r"$x / \lambda_D$", fontsize=25)
        ax.set_ylabel(r"$y / \lambda_D$", fontsize=25)
        bar.set_label(r"$\phi$", fontsize=25)
        if show_positions:
            ax.scatter(positions[:, 0], positions[:, 1], s=5, c='r', marker='o')
        ax.set_aspect("equal")
        plt.show()


class TestField(unittest.TestCase):
    def test_data(self):
        N, L, n = load_config(r"electrosctatic\sim_two_stream.json")
        delta_r = L / n
        phi = load_rho(r"electrosctatic\phi\step_0_.dat", n, delta_r)
        expected_field = load_field(r"electrosctatic\Efield\step_0_.dat", n, delta_r)
        field = field_nodes(phi, n, delta_r)
        np.testing.assert_allclose(field, expected_field, atol=1.e-05)

    def test_field_particles(self):
        Lx = Ly = n_x = n_y = 3
        dx = dy = 1
        delta_r = np.array([dx, dy])
        n = np.array([n_x, n_y])
        positions = np.array([[0.5, 0.5], [0.5, 0.5], [1.5, 1.5]])
        field = np.zeros([Lx, Ly, 3])
        field[0, 0] = np.array([0, 1, 0])
        field[1, 2] = np.array([0, -1, 0])
        E = field_particles(field, positions, n, delta_r)
        np.testing.assert_allclose(E, np.array([[0., 0.25, 0.],
                                                [0., 0.25, 0.],
                                                [0., -0.25, 0.]]))


class TestPhaseSpace(unittest.TestCase):
    def test_data(self):
        dt = 0.1
        B = np.array([0, 0, 0])
        N, L, n = load_config(r"electrosctatic\sim_two_stream.json")
        delta_r = L / n
        positions, velocities, q_m, moves = local_initial_state(r"electrosctatic\two_stream.dat")
        expected_field = load_field(r"electrosctatic\Efield\step_0_.dat", n, delta_r)
        expected_positions, expected_velocities = load_space(
            r"electrosctatic\phase_space\step_0_.dat")
        e_field_p = field_particles(expected_field, positions, n, delta_r)
        velocities = boris(velocities, q_m, e_field_p, B, -0.5 * dt)
        positions, velocities = update(positions, velocities, q_m, e_field_p, B, L, dt)
        velocities = boris(velocities, q_m, e_field_p, B, 0.5 * dt)
        np.testing.assert_allclose(positions[moves == 1], expected_positions, atol=5e-05)
        np.testing.assert_allclose(velocities[moves == 1], expected_velocities, atol=5.01835383e-06)


class TestEnergy(unittest.TestCase):
    def test_data(self):
        N, L, n = load_config(r"electrosctatic\sim_two_stream.json")
        delta_r = L / n
        step_re = re.compile(r"\\step_(\d+)_.dat$")
        expected_energies = load_energy(r"electrosctatic\energy\energy.dat")
        fe = []
        efe = []
        for phase_file, rho_file, phi_file in zip(glob(r"electrosctatic\phase_space\step_*_.dat"),
                                                  glob(r"electrosctatic\rho\step_*_.dat"),
                                                  glob(r"electrosctatic\phi\step_*_.dat")):
            step = int(step_re.search(phase_file).group(1))
            positions, velocities = load_space(phase_file)
            mass = (L[0] * L[1] * 1) / N
            kinetic_energy = calculate_kinetic_energy(velocities, mass)
            np.testing.assert_allclose(kinetic_energy, expected_energies[step][0], rtol=0.002)

            rho = load_rho(rho_file, n, delta_r)
            phi = load_rho(phi_file, n, delta_r)
            field_energy = (rho * phi).sum() * 0.5
            np.testing.assert_allclose(field_energy, expected_energies[step][1], atol=0.004)

if __name__ == '__main__':
    unittest.main()
