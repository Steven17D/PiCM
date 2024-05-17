import numpy as np


def local_initial_state(file_name):
    positions_text = open(file_name, "r").readlines()
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
