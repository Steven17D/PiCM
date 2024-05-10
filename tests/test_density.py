import unittest

import numpy as np
from matplotlib import pyplot as plt

from main import density

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

    def test_multiple_particles_field(self):
        Lx = Ly = 64
        dx = dy = 1
        x, y = np.meshgrid(np.arange(0, Lx), np.arange(0, Ly))
        positions = np.random.uniform(0, Ly, (100, 2))
        rho = density(positions, np.array([Lx, Ly]), np.array([1., 1.]), 1.0, 1.0)
        fig, ax = plt.subplots()
        ax.title.set_text(r"$\omega_{\rm{pe}}$")
        color_map = ax.pcolormesh(x, y, rho, shading="gouraud", cmap="jet")
        bar = plt.colorbar(color_map, cax=ax)
        ax.set_xlim(0, Lx - dx)
        ax.set_ylim(0, Ly - dy)
        ax.set_xlabel(r"$x / \lambda_D$", fontsize=25)
        ax.set_ylabel(r"$y / \lambda_D$", fontsize=25)
        bar.set_label(r"$\rho$", fontsize=25)
        ax.scatter(positions[:, 0], positions[:, 1], s=5, c='r', marker='o')
        ax.set_aspect("equal")
        plt.show()



if __name__ == '__main__':
    unittest.main()
