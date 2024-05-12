import unittest

import numpy as np
from matplotlib import pyplot as plt

from main import density, boris, field_nodes, field_particles, potential, update


def local_initial_state():
    positions_file = r"C:\Projects\PiCM-cpp\electrosctatic\two_stream.dat"
    positions_text = open(positions_file, "r").readlines()
    positions = np.empty((len(positions_text), 2))
    velocities = np.empty((len(positions_text), 3))
    charges = np.empty((len(positions_text),))
    moves = np.empty((len(positions_text),))
    for p, line in enumerate(positions_text):
        x, y, vx, vy, vz, qm, m = [float(d) for d in line.split(" ")]
        positions[p] = np.array([x, y])
        velocities[p] = np.array([vx, vy, vz])
        charges[p] = qm
        moves[p] = m
    return positions, velocities, charges, moves


def load_rho(file_name):
    expected_data_text = open(file_name, "r").readlines()
    expected_rho = np.empty(shape=(64, 64), dtype=float)
    for line in expected_data_text:
        i, j, value = [float(d) for d in line.split(" ")]
        expected_rho[int(i), int(j)] = value
    return expected_rho


def load_field(file_name):
    expected_data_text = open(file_name, "r").readlines()
    data = np.empty(shape=(64, 64, 3), dtype=float)
    for line in expected_data_text:
        i, j, *value = [float(d) for d in line.split(" ")]
        data[int(i), int(j)] = np.array([*value, 0])
    return data


def load_space(file_name):
    lines = open(file_name, "r").readlines()
    N = len(lines)
    positions = np.empty(shape=(N, 2), dtype=float)
    velocities = np.empty(shape=(N, 3), dtype=float)
    for p, line in enumerate(lines, start=0):
        x, y, vx, vy, vz = [float(d) for d in line.split(" ")]
        positions[p] = np.array([x, y])
        velocities[p] = np.array([vx, vy, vz])
    return positions, velocities


class TestDensity(unittest.TestCase):
    def test_single_particle(self):
        n = np.array([3, 3])
        positions = np.array([[0.5, 0.5]])
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
        expected = np.array([
            [0.25, 0.25, 0],
            [0.25, 0.25, 0],
            [0, 0, 0],
        ])
        np.testing.assert_allclose(rho, expected)

    def test_single_particle_origin(self):
        n = np.array([3, 3])
        positions = np.array([[0, 0]])
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
        expected = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_two_particles_origin(self):
        n = np.array([3, 3])
        positions = np.array([[0, 0], [0, 0]])
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
        expected = np.array([
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_two_particles_same_cell(self):
        n = np.array([3, 3])
        positions = np.array([[0.5, 0.5], [0.5, 0.5]])
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
        expected = np.array([
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0, 0, 0],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_single_particle_center(self):
        n = np.array([3, 3])
        positions = np.array([[1.5, 1.5]])
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
        expected = np.array([
            [0, 0, 0],
            [0, 0.25, 0.25],
            [0, 0.25, 0.25],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_single_particle_edge(self):
        n = np.array([3, 3])
        positions = np.array([[2, 2]])
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
        expected = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]).astype(np.float64)
        np.testing.assert_allclose(rho, expected)

    def test_two_particles_edges(self):
        n = np.array([3, 3])
        positions = np.array([[0, 0], [2, 2]])
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
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
        rho = density(positions, n, np.array([1., 1.]), 1.0, 1.0)
        self.assertEqual(rho.sum(), N)

    def test_multiple_particles_density(self):
        Lx = Ly = 64
        dx = dy = 1
        # x, y = np.meshgrid(np.arange(0, Lx), np.arange(0, Ly))
        positions = np.random.uniform(0, Ly, (10 ** 5, 2))
        rho = density(positions, np.array([Lx, Ly]), np.array([1., 1.]), 1.0, 1.0)
        plt.imshow(rho)
        # fig, ax = plt.subplots()
        # ax.title.set_text(r"$\omega_{\rm{pe}}$")
        color_map = plt.pcolormesh(rho, shading="gouraud", cmap="jet")
        bar = plt.colorbar(color_map)
        plt.xlim(0, Lx - dx)
        plt.ylim(0, Ly - dy)
        plt.xlabel(r"$x / \lambda_D$", fontsize=25)
        plt.ylabel(r"$y / \lambda_D$", fontsize=25)
        bar.set_label(r"$\rho$", fontsize=25)
        # ax.scatter(positions[:, 0], positions[:, 1], s=5, c='r', marker='o')
        # ax.set_aspect("equal")
        plt.show()

    def test_cpp(self):
        positions, _, charges, _ = local_initial_state()
        expected_rho = load_rho(r"C:\Projects\PiCM-cpp\electrosctatic\rho\step_0_.dat")
        Lx = Ly = 64
        rho = density(positions, charges, np.array([Lx, Ly]), np.array([1., 1.]), 1.0, 1.0)
        np.testing.assert_allclose(rho, expected_rho, rtol=1e-5)


class TestPotential(unittest.TestCase):

    def test_cpp(self):
        rho = load_rho(r"C:\Projects\PiCM-cpp\electrosctatic\rho\step_0_.dat")
        expected_phi = load_rho(r"C:\Projects\PiCM-cpp\electrosctatic\phi\step_0_.dat")
        n = np.array([64, 64])
        phi = potential(rho, n, np.array([1., 1.]))
        np.testing.assert_allclose(phi, expected_phi, rtol=0.00275092)

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
        rho = density(positions, np.array([Lx, Ly]), delta_r, 1.0, 1.0)
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
        Lx = Ly = n_x = n_y = 64
        dx = dy = 1
        delta_r = np.array([dx, dy])
        n = np.array([n_x, n_y])

        phi = load_rho(r"C:\Projects\PiCM-cpp\electrosctatic\phi\step_0_.dat")
        expected_field = load_field(r"C:\Projects\PiCM-cpp\electrosctatic\Efield\step_0_.dat")
        field = field_nodes(phi, n, delta_r)
        np.testing.assert_allclose(field, expected_field, atol=4.8e-06)

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
        Lx = Ly = n_x = n_y = 64
        L = np.array([Lx, Ly])
        dx = dy = 1
        delta_r = np.array([dx, dy])
        n = np.array([n_x, n_y])
        dt = 0.1
        B = np.array([0, 0, 0])
        positions, velocities, charges, moves = local_initial_state()
        expected_field = load_field(r"C:\Projects\PiCM-cpp\electrosctatic\Efield\step_0_.dat")
        expected_positions, expected_velocities = load_space(
            r"C:\Projects\PiCM-cpp\electrosctatic\phase_space\step_0_.dat")
        e_field_p = field_particles(expected_field, positions, moves, n, delta_r)
        velocities = boris(velocities, charges, moves, e_field_p, B, -0.5 * dt)
        positions, velocities = update(positions, velocities, charges, moves, e_field_p, B, L, dt)
        velocities = boris(velocities, charges, moves, e_field_p, B, 0.5 * dt)
        np.testing.assert_allclose(positions[moves == 1], expected_positions, atol=5e-05)
        np.testing.assert_allclose(velocities[moves == 1], expected_velocities, atol=5.01835383e-06)


if __name__ == '__main__':
    unittest.main()
