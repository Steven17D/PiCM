import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from PiCM.loader import local_initial_state, load_rho, load_field, load_space, load_energy, load_config
from PiCM.main import calculate_kinetic_energy
from PiCM.simulation import density, boris, field_nodes, field_particles, potential, update


class TestDensity(unittest.TestCase):
    def test_single_particle(self):
        n = np.array([3, 3])
        positions = np.array([[0.5, 0.5]])
        charges = np.array([1])
        rho = density(positions, charges, n, np.array([1., 1.]))
        expected = np.array([
            [0.25, 0.25, 0],
            [0.25, 0.25, 0],
            [0, 0, 0],
        ])
        np.testing.assert_allclose(rho, expected)

    def test_single_particle_origin(self):
        n = np.array([3, 3])
        positions = np.array([[0, 0]])
        charges = np.array([1])
        rho = density(positions, charges, n, np.array([1., 1.]))
        expected = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_two_particles_origin(self):
        n = np.array([3, 3])
        positions = np.array([[0, 0], [0, 0]])
        charges = np.ones(2)
        rho = density(positions, charges, n, np.array([1., 1.]))
        expected = np.array([
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_two_particles_same_cell(self):
        n = np.array([3, 3])
        positions = np.array([[0.5, 0.5], [0.5, 0.5]])
        charges = np.ones(2)
        rho = density(positions, charges, n, np.array([1., 1.]))
        expected = np.array([
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0, 0, 0],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_single_particle_center(self):
        n = np.array([3, 3])
        positions = np.array([[1.5, 1.5]])
        charges = np.ones(1)
        rho = density(positions, charges, n, np.array([1., 1.]))
        expected = np.array([
            [0, 0, 0],
            [0, 0.25, 0.25],
            [0, 0.25, 0.25],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_single_particle_edge(self):
        n = np.array([3, 3])
        positions = np.array([[2, 2]])
        charges = np.ones(1)
        rho = density(positions, charges, n, np.array([1., 1.]))
        expected = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_two_particles_edges(self):
        n = np.array([3, 3])
        positions = np.array([[0, 0], [2, 2]])
        charges = np.ones(2)
        rho = density(positions, charges, n, np.array([1., 1.]))
        expected = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_multiple_particles_total_charge(self):
        n = np.array([3, 3])
        N = np.random.randint(0, 100)
        positions = np.random.uniform(0, 3, (N, 2))
        charges = np.ones(N)
        rho = density(positions, charges, n, np.array([1., 1.]))
        self.assertEqual(rho.sum(), N)

    def test_multiple_particles_density(self):
        Lx = Ly = 64
        dx = dy = 1
        N = 10 ** 5
        positions = np.random.uniform(0, Ly, (N, 2))
        charges = np.ones(N)
        rho = density(positions, charges, np.array([Lx, Ly]), np.array([1., 1.]))
        plt.imshow(rho)
        color_map = plt.pcolormesh(rho, shading="gouraud", cmap="jet")
        bar = plt.colorbar(color_map)
        plt.xlim(0, Lx - dx)
        plt.ylim(0, Ly - dy)
        plt.xlabel(r"$x / \lambda_D$", fontsize=25)
        plt.ylabel(r"$y / \lambda_D$", fontsize=25)
        bar.set_label(r"$\rho$", fontsize=25)
        plt.show()

    def test_data(self):
        N, L, n = load_config(r"electrosctatic\sim_two_stream.json")
        delta_r = L / n
        positions, _, q_m, _ = local_initial_state(r"electrosctatic\two_stream.dat")
        expected_rho = load_rho(r"electrosctatic\rho\first_rho0.dat", n, delta_r)
        charges = (L[0] * L[1] * q_m) / N
        # rho = density(positions, charges, n, delta_r)
        split = positions.shape[0] // 10000
        rho = np.sum([density(p, c, n, delta_r) for p, c in zip(
            np.split(positions, split),
            np.split(charges, split))], axis=0)
        np.testing.assert_allclose(rho, expected_rho, rtol=1e-5)


class TestPotential(unittest.TestCase):

    def test_data(self):
        N, L, n = load_config(r"electrosctatic\sim_two_stream.json")
        delta_r = L / n
        rho = load_rho(r"electrosctatic\rho\first_rho0.dat", n, delta_r)
        expected_phi = load_rho(r"electrosctatic\phi\step_0_.dat", n, delta_r)
        phi = potential(rho, n, delta_r)
        np.testing.assert_allclose(phi, expected_phi, rtol=0.008)

    def test_single_particles_phi(self):
        self._plot_phi(1, show_positions=True)

    def test_multiple_particles_phi(self):
        N = 10 ** 1
        self._plot_phi(N, show_positions=True)

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
        color_map = ax.pcolormesh(x, y, phi, shading="gouraud", cmap="jet")
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
        np.testing.assert_allclose(field, expected_field, atol=5e-06)

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
