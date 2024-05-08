"""
We refer to the superparticle as particle
"""
import numpy as np
import scipy.constants as const
from acceptance_rejection import get_random_value
from scipy.stats import truncnorm, maxwell, norm


k_b = const.physical_constants['Boltzmann constant'][0]
omega_p = np.sqrt(n_alpha * (q_alpha ** 2) / (m_alpha))
v_th = np.array([1, 1]) * np.sqrt(k_b * T_alpha * m_alpha)  # Thermal velocity
debye_length = v_th / omega_p
L = np.array([64, 64]) * debye_length  # Lx and Ly are the lengths of the system in the x and y directions in units of the Debye length.
n = np.array([64, 64])
delta_r = L / n  # Vector of delta x and delta y
dxdy = np.multiply.reduce(delta_r)
q = 1  # Charge of a cell? TODO: Check
ro_c = q / dxdy
assert 0.5 * delta_r[0] < debye_length
assert 0.5 * delta_r[1] < debye_length

N = 10 ** 6
v_d = 5 * v_th
n_e = N / np.multiply.reduce(L)


def maxwell_distribution(v):
    return (
        (n_e / np.sqrt(2 * np.pi * (v_th ** 2))) * (
            np.exp((-(v_th - v_d) ** 2)/(2 * (v_th ** 2))) + np.exp((-(v_th + v_d) ** 2)/(2 * (v_th ** 2)))
        )
    )


class Particle:
    r: np.ndarray  # 2D
    v: np.ndarray  # 3D

    def __init__(self, r, v):
        self.r = np.random.uniform(np.zeros_like(L), L)
        get_random_value(maxwell_distribution, 0, 100, )
        # maxwell.
        # self.v = np.array(, )


def main():
    particles: list[Particle] = []

    ro = np.zeros(n)
    for particle in particles:
        i, j = np.floor(particle.r / delta_r)
        h = particle.r - np.array([i, j]) * delta_r
        h_n = delta_r - h
        # Equations 6a-6d
        ro[i, j] = ro_c * (np.multiply.reduce(h_n) / dxdy)
        ro[i, (j+1) % n[1]] = ro_c * (h_n[0] * h[1] / dxdy)
        ro[(i+1) % n[0], j] = ro_c * (h_n[1] * h[0] / dxdy)
        ro[(i+1) % n[0], (j+1) % n[1]] = ro_c * (np.multiply.reduce(h) / dxdy)


if __name__ == '__main__':
    main()
