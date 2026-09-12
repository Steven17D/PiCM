"""
Implementation of PiCM simulation.
"""
import numpy as np
from tqdm import tqdm


def density(positions: np.ndarray, charges: np.array, n: np.array, delta_r: np.array):
    """
    Cloud-in-cell charge density on a periodic grid of shape (nx, ny).
    """
    rho = np.zeros(n, dtype=np.float64)
    if positions.shape[0] == 0:
        return rho

    nx, ny = n[0], n[1]
    cell_area = delta_r[0] * delta_r[1]
    fx = positions[:, 0] / delta_r[0]
    fy = positions[:, 1] / delta_r[1]
    i = np.floor(fx).astype(int)
    j = np.floor(fy).astype(int)
    wx = fx - i
    wy = fy - j
    wx0 = 1.0 - wx
    wy0 = 1.0 - wy
    i0 = i % nx
    j0 = j % ny
    i1 = (i + 1) % nx
    j1 = (j + 1) % ny
    q = charges / cell_area
    np.add.at(rho, (i0, j0), wx0 * wy0 * q)
    np.add.at(rho, (i0, j1), wx0 * wy * q)
    np.add.at(rho, (i1, j0), wx * wy0 * q)
    np.add.at(rho, (i1, j1), wx * wy * q)
    return rho


def potential(rho: np.ndarray, n: np.array, delta_r: np.array) -> np.ndarray:
    """
    Calculate the potential (phi) from the charge density (rho)
    :param rho: Grid of charge density
    :param n: Grid dimensions
    :param delta_r: Grid cell size
    :return: Potential of charge density in grid form
    """
    nx, ny = int(n[0]), int(n[1])
    dx2, dy2 = delta_r[0] ** 2, delta_r[1] ** 2
    sx = np.sin(np.pi * np.arange(nx) / nx)
    sy = np.sin(np.pi * np.arange(ny) / ny)
    # 5-point FD eigenvalues: 4 sin^2(pi k / n) / dr^2. Cross-multiplied
    # rectangular weights match 2 - W - 1/W without Wm recurrence roundoff.
    lap = 4.0 * sx[:, None] ** 2 / dx2 + 4.0 * sy[None, :] ** 2 / dy2
    lap[0, 0] = 1.0
    phi_k = np.fft.fft2(rho) / lap
    phi_k[0, 0] = 0.0
    return np.real(np.fft.ifft2(phi_k))


def field_nodes(phi: np.ndarray, n: np.array, delta_r: np.array) -> np.ndarray:
    """
    Calculate the electric field
    :param rho: Grid of charge density
    :param n: Grid dimensions
    :param delta_r: Grid cell size
    :return: Electric field in grid form
    """
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


def field_particles(field: np.ndarray, positions: np.ndarray, n: np.array, delta_r: np.array) -> np.ndarray:
    """
    Calculate the electric field exerted on each particle.
    :param field: Electric field in grid form
    :param positions: List of charge positions
    :param n: Grid dimensions
    :param delta_r: Grid cell size
    :return: Electric field exerted on each particle
    """
    dx, dy = delta_r
    ijs = np.floor(positions / delta_r).astype(int)
    h = positions - ijs * delta_r
    nxt_ijs = (ijs + 1) % n
    A = ((dx - h[:, 0]) * (dy - h[:, 1]))[:, np.newaxis] * field[ijs[:, 0], ijs[:, 1]]
    B = ((dx - h[:, 0]) * h[:, 1])[:, np.newaxis] * field[ijs[:, 0], nxt_ijs[:, 1]]
    C = (h[:, 0] * (dy - h[:, 1]))[:, np.newaxis] * field[nxt_ijs[:, 0], ijs[:, 1]]
    D = (h[:, 0] * h[:, 1])[:, np.newaxis] * field[nxt_ijs[:, 0], nxt_ijs[:, 1]]
    E = A + B + C + D
    return E / (dx * dy)


def boris(velocities: np.ndarray, q_m: np.array, E: np.ndarray, B: np.array, dt: np.float64):
    """
    Calculate new velocities of particles in an electro-magnetic field using boris algorithm.
    :param velocities: Velocities of particles
    :param q_m: Charge to mass ration of particles
    :param E: Electric field
    :param B: Magnetic field
    :param dt: Time step
    :return: Velocities of particles
    """
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
    """
    Generate simulation steps.
    :param positions: Positions of particles
    :param velocities: Velocities of particles
    :param q_m: Charge to mass ration of particles
    :param charges: Charges of particles
    :param moves: Move attribute of particles
    :param L: System dimensions
    :param n: Grid dimensions
    :param delta_r: Grid cell size
    :param B: Magnetic field
    :param dt: Time step
    :param steps: Number of simulation steps
    """
    statics = moves == 0
    # Calculate the static charge density grid
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


def calculate_kinetic_energy(velocities, masses):
    return (masses * (velocities[:, 0] ** 2 + velocities[:, 1] ** 2)).sum() / 2


def calculate_field_energy(rho, phi, delta_r):
    return 0.5 * np.sum(rho * phi) * np.prod(delta_r)
