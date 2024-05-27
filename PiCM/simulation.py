"""
Implementation of PiCM simulation.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


grhos: np.ndarray = None


def _density(positions: np.ndarray, charges: np.array, i: int, n: np.array, delta_r: np.array) -> np.ndarray:
    """
    Calculate the grid of charge density
    :param positions: List of charge positions
    :param charges: List of charges
    :param i: Index of global rho buffer
    :param n: Grid dimensions
    :param delta_r: Grid cell size
    :return: Grid of charge density
    """
    global grhos
    rho = np.zeros(n)
    ijs = np.floor(positions / delta_r).astype(int)
    ijs_up = (ijs + [0, 1]) % n
    ijs_right = (ijs + [1, 0]) % n
    ijs_diag = (ijs + [1, 1]) % n
    h = positions - ijs * delta_r
    h_n = delta_r - h

    rhos = grhos[i]

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


def density(positions: np.ndarray, charges: np.array, n: np.array, delta_r: np.array):
    """
    Calculate the charge density in parallel.
    """
    global grhos

    thread_count = os.cpu_count()
    chunk_size = positions.shape[0] // thread_count
    if grhos is None:
        grhos = np.zeros((thread_count, chunk_size, *n))

    remainder = positions.shape[0] % thread_count
    remainder_positions = positions[-remainder:]
    remainder_charges = charges[-remainder:]
    remainder_rho = _density(remainder_positions, remainder_charges, 0, n, delta_r)

    with ThreadPoolExecutor(max_workers=thread_count) as e:
        results = e.map(lambda p: _density(p[0], p[1], p[2], n, delta_r),
                        zip(np.split(positions[:-remainder], thread_count),
                            np.split(charges[:-remainder], thread_count),
                            range(thread_count))
                        )
        return np.sum(results, axis=0) + remainder_rho


def potential(rho: np.ndarray, n: np.array, delta_r: np.array) -> np.ndarray:
    """
    Calculate the potential (phi) from the charge density (rho)
    :param rho: Grid of charge density
    :param n: Grid dimensions
    :param delta_r: Grid cell size
    :return: Potential of charge density in grid form
    """
    rho = rho.astype(complex)

    # FFT rho to rho_k
    for xi in range(n[0]):
        rho[xi, :] = np.fft.fft(rho[xi, :])
    for yi in range(n[1]):
        rho[:, yi] = np.fft.fft(rho[:, yi])

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
        phi_k[xi, :] = np.fft.ifft(phi_k[xi, :])
    for yi in range(n[1]):
        phi_k[:, yi] = np.fft.ifft(phi_k[:, yi])

    return np.real(phi_k)


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
