import unittest
from pathlib import Path

import numpy as np

from PiCM.loader import local_initial_state, load_rho, load_field, load_space, load_energy, load_config
from PiCM.simulation import density, boris, field_nodes, field_particles, potential, update, calculate_kinetic_energy, calculate_field_energy, simulate


_FIXTURE_DIR = Path(__file__).resolve().parent / "electrosctatic"


def _fixture(*parts):
    return _FIXTURE_DIR.joinpath(*parts)


def _step_index(path):
    return int(path.stem.split("_")[1])


def _fixture_steps(subdir):
    files = {_step_index(path): path for path in _FIXTURE_DIR.glob(f"{subdir}/step_*_.dat")}
    if not files:
        raise AssertionError(f"no step_*.dat fixtures under {_FIXTURE_DIR / subdir}")
    return files


class TestDensity(unittest.TestCase):
    def test_data(self):
        N, L, n = load_config(_fixture("sim_two_stream.json"))
        delta_r = L / n
        positions, _, q_m, _ = local_initial_state(_fixture("two_stream.dat"))
        expected_rho = load_rho(_fixture("rho", "step_0_.dat"), n, delta_r)
        charges = (L[0] * L[1] * q_m) / N
        rho = density(positions, charges, n, delta_r)
        np.testing.assert_allclose(rho, expected_rho, rtol=1e-5)

    def test_same_process_grid_resize(self):
        L = np.array([4.0, 3.0])
        positions = np.array([[0.25, 0.25], [1.1, 0.8], [3.9, 2.9], [2.0, 1.5]])
        charges = np.array([1.0, -0.5, 2.0, -0.25])
        total_charge = charges.sum()
        for n in (np.array([8, 8]), np.array([5, 7]), np.array([16, 10])):
            delta_r = L / n
            rho = density(positions, charges, n, delta_r)
            self.assertEqual(rho.shape, tuple(n))
            np.testing.assert_allclose(rho.sum() * np.prod(delta_r), total_charge, atol=1e-12)

    def test_rectangular_deposition_charge_and_corners(self):
        n = np.array([3, 2])
        delta_r = np.array([2.0, 0.5])
        charges = np.array([2.0])
        interior = np.array([[0.5, 0.125]])
        rho = density(interior, charges, n, delta_r)
        self.assertEqual(rho.shape, (3, 2))
        expected = np.zeros((3, 2))
        expected[0, 0] = 1.125
        expected[0, 1] = 0.375
        expected[1, 0] = 0.375
        expected[1, 1] = 0.125
        np.testing.assert_allclose(rho, expected)

        domain = n * delta_r
        corner = np.array([[domain[0] - 0.5, domain[1] - 0.125]])
        wrapped = density(corner, charges, n, delta_r)
        expected_corner = np.zeros((3, 2))
        expected_corner[2, 1] = 0.125
        expected_corner[0, 1] = 0.375
        expected_corner[2, 0] = 0.375
        expected_corner[0, 0] = 1.125
        np.testing.assert_allclose(wrapped, expected_corner)


class TestPotential(unittest.TestCase):

    def test_data(self):
        N, L, n = load_config(_fixture("sim_two_stream.json"))
        delta_r = L / n
        rho = load_rho(_fixture("rho", "step_0_.dat"), n, delta_r)
        expected_phi = load_rho(_fixture("phi", "step_0_.dat"), n, delta_r)
        phi = potential(rho, n, delta_r)
        np.testing.assert_allclose(phi, expected_phi, rtol=0.008)

    def test_zero_mean_poisson_rectangular_cells(self):
        n = np.array([8, 12])
        delta_r = np.array([0.25, 0.5])
        x = np.arange(n[0])[:, None]
        y = np.arange(n[1])[None, :]
        rho = np.sin(2.0 * np.pi * x / n[0]) * np.cos(4.0 * np.pi * y / n[1])
        rho = rho - rho.mean()
        phi = potential(rho, n, delta_r)
        np.testing.assert_allclose(phi.mean(), 0.0, atol=1e-12)
        dx, dy = delta_r
        laplacian = (
            (np.roll(phi, -1, axis=0) - 2.0 * phi + np.roll(phi, 1, axis=0)) / dx ** 2
            + (np.roll(phi, -1, axis=1) - 2.0 * phi + np.roll(phi, 1, axis=1)) / dy ** 2
        )
        np.testing.assert_allclose(laplacian, -rho, rtol=1e-10, atol=1e-12)


class TestField(unittest.TestCase):
    def test_data(self):
        N, L, n = load_config(_fixture("sim_two_stream.json"))
        delta_r = L / n
        phi = load_rho(_fixture("phi", "step_0_.dat"), n, delta_r)
        expected_field = load_field(_fixture("Efield", "step_0_.dat"), n, delta_r)
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
        N, L, n = load_config(_fixture("sim_two_stream.json"))
        delta_r = L / n
        positions, velocities, q_m, moves = local_initial_state(_fixture("two_stream.dat"))
        expected_field = load_field(_fixture("Efield", "step_0_.dat"), n, delta_r)
        expected_positions, expected_velocities = load_space(
            _fixture("phase_space", "step_0_.dat"))
        e_field_p = field_particles(expected_field, positions, n, delta_r)
        velocities = boris(velocities, q_m, e_field_p, B, -0.5 * dt)
        positions, velocities = update(positions, velocities, q_m, e_field_p, B, L, dt)
        velocities = boris(velocities, q_m, e_field_p, B, 0.5 * dt)
        np.testing.assert_allclose(positions[moves == 1], expected_positions, atol=5e-05)
        np.testing.assert_allclose(velocities[moves == 1], expected_velocities, atol=5.01835383e-06)


class TestEnergy(unittest.TestCase):
    def test_data(self):
        N, L, n = load_config(_fixture("sim_two_stream.json"))
        delta_r = L / n
        expected_energies = load_energy(_fixture("energy", "energy.dat"))
        phase_files = _fixture_steps("phase_space")
        rho_files = _fixture_steps("rho")
        phi_files = _fixture_steps("phi")
        self.assertEqual(set(phase_files), set(rho_files))
        self.assertEqual(set(phase_files), set(phi_files))
        # energy.dat field-energy values are raw node sums 0.5*sum(rho*phi).
        # Physical energy integrates that density over cell area, so the
        # reference is multiplied by prod(delta_r) when comparing the helper.
        cell_area = np.prod(delta_r)
        for step in sorted(phase_files):
            _, velocities = load_space(phase_files[step])
            mass = (L[0] * L[1] * 1) / N
            kinetic_energy = calculate_kinetic_energy(velocities, mass)
            np.testing.assert_allclose(kinetic_energy, expected_energies[step][0], rtol=0.002)

            rho = load_rho(rho_files[step], n, delta_r)
            phi = load_rho(phi_files[step], n, delta_r)
            field_energy = calculate_field_energy(rho, phi, delta_r)
            np.testing.assert_allclose(
                field_energy,
                expected_energies[step][1] * cell_area,
                atol=0.004 * cell_area,
            )

    def test_field_energy_physical_grid_independence(self):
        L = np.array([4.0, 6.0])
        rho_val, phi_val = 2.0, -3.0
        expected = 0.5 * rho_val * phi_val * np.prod(L)
        for n in (np.array([8, 12]), np.array([16, 24]), np.array([4, 6])):
            delta_r = L / n
            rho = np.full(tuple(n), rho_val)
            phi = np.full(tuple(n), phi_val)
            np.testing.assert_allclose(
                calculate_field_energy(rho, phi, delta_r), expected
            )

    def test_two_stream_energy_evolution_across_resolution(self):
        _, L, _ = load_config(_fixture("sim_two_stream.json"))
        state = local_initial_state(_fixture("two_stream.dat"))
        positions, velocities, q_m, moves = (values[::50] for values in state)
        N = len(positions)
        charges = np.prod(L) * q_m / N
        mass = np.prod(L) / N
        for n_cells in (64, 128, 256):
            with self.subTest(resolution=n_cells):
                n = np.array([n_cells, n_cells])
                delta_r = L / n
                total = []
                for _, vel, rho, phi, _, _ in simulate(
                    positions, velocities, q_m, charges, moves,
                    L, n, delta_r, np.zeros(3), 0.1, 31
                ):
                    total.append(
                        calculate_kinetic_energy(vel, mass)
                        + calculate_field_energy(rho, phi, delta_r)
                    )
                total = np.asarray(total)
                self.assertLess(np.max(np.abs(total / total[0] - 1)), 0.01)


if __name__ == '__main__':
    unittest.main()
