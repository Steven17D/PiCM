import unittest

import numpy as np

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



if __name__ == '__main__':
    unittest.main()
