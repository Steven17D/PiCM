import numpy as np


def read_lines(file_name):
    with open(file_name) as f:
        return f.readlines()


def load_rho(file_name):
    expected_rho = np.empty(shape=(64, 64), dtype=float)
    for line in read_lines(file_name):
        i, j, value = [float(d) for d in line.split(" ")]
        expected_rho[int(i), int(j)] = value
    return expected_rho


def load_field(file_name):
    data = np.empty(shape=(64, 64, 3), dtype=float)
    for line in read_lines(file_name):
        i, j, *value = [float(d) for d in line.split(" ")]
        data[int(i), int(j)] = np.array([*value, 0])
    return data


def load_space(file_name):
    lines = read_lines(file_name)
    N = len(lines)
    positions = np.empty(shape=(N, 2), dtype=float)
    velocities = np.empty(shape=(N, 3), dtype=float)
    for p, line in enumerate(lines, start=0):
        x, y, vx, vy, vz = [float(d) for d in line.split(" ")]
        positions[p] = np.array([x, y])
        velocities[p] = np.array([vx, vy, vz])
    return positions, velocities


def load_energy(file_name):
    r = {}
    for line in read_lines(file_name):
        step, KE, FE = [float(d) for d in line.split(" ")]
        r[int(step)] = [KE, FE]
    return r


def local_initial_state(file_name):
    positions_text = read_lines(file_name)
    positions = np.empty((len(positions_text), 2))
    velocities = np.empty((len(positions_text), 3))
    qms = np.empty((len(positions_text),))
    moves = np.empty((len(positions_text),))
    for p, line in enumerate(positions_text):
        x, y, vx, vy, vz, qm, m = [float(d) for d in line.split(" ")]
        positions[p] = np.array([x, y])
        velocities[p] = np.array([vx, vy, vz])
        qms[p] = qm
        moves[p] = m
    return positions, velocities, qms, moves
