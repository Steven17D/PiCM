import numpy as np
from tqdm import tqdm


omega_pe = 1


def fft(v: np.ndarray) -> np.ndarray:
    return np.fft.fft(v)


def ifft(v: np.ndarray) -> np.ndarray:
    return np.fft.ifft(v)


rhos = None  # Rhos are initialized once for performance


def density(positions: np.ndarray, charges, n, delta_r):
    """
    Calculate the density of particles
    :param positions:
    :return:
    """
    global rhos
    rho = np.zeros(n)
    ijs = np.floor(positions / delta_r).astype(int)  # TODO: Check round of floor
    ijs_up = (ijs + [0, 1]) % n
    ijs_right = (ijs + [1, 0]) % n
    ijs_diag = (ijs + [1, 1]) % n
    h = positions - ijs * delta_r
    h_n = delta_r - h
    if rhos is None or rhos.shape[1:] != rho.shape:
        rhos = np.full((positions.shape[0], *rho.shape), rho)

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


def potential(rho: np.ndarray, n, delta_r):
    """
    Calculate the potential (phi) from the charge density (rho)
    TODO: Write the proof of the FFT from rho to phi
    :param rho:
    :return:
    """
    rho = rho.astype(complex)

    # FFT rho to rho_k
    for xi in range(n[0]):
        rho[xi, :] = fft(rho[xi, :])
    for yi in range(n[1]):
        rho[:, yi] = fft(rho[:, yi])

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
        phi_k[xi, :] = ifft(phi_k[xi, :])
    for yi in range(n[1]):
        phi_k[:, yi] = ifft(phi_k[:, yi])

    return np.real(phi_k)


def field_nodes(phi: np.ndarray, n, delta_r):
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


def field_particles(field: np.ndarray, positions: np.array, n, delta_r):
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


def boris(velocities, q_m, E, B, dt):
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
    statics = moves == 0
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
