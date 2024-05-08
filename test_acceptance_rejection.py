import unittest

import numpy as np
from matplotlib import pyplot as plt
from acceptance_rejection import get_random_value

n_e = 0.5
v_th = 1
v_d = 5
def maxwell_distribution(v):
    return (
        (n_e / np.sqrt(2 * np.pi * (v_th ** 2))) * (
            np.exp((-(v - v_d) ** 2)/(2 * (v_th ** 2))) + np.exp((-(v + v_d) ** 2)/(2 * (v_th ** 2)))
        )
    )

class MyTestCase(unittest.TestCase):
    def test_get_random_value(self):
        f = lambda x: ((3 / 8) * (1 + x ** 2))
        results = [get_random_value(f, -1, 1, 3 / 4) for _ in range(100000)]
        xs = np.linspace(-1, 1, 1000)
        ys = f(xs)
        fig, ax = plt.subplots()
        ax.hist(results, bins=100, density=True)
        ax.plot(xs, ys)
        plt.show()

    def test_mb(self):
        min_value, max_value = -10, 10
        results = [get_random_value(maxwell_distribution, min_value, max_value, v_d) for _ in range(10000)]
        xs = np.linspace(min_value, max_value, 1000)
        ys = maxwell_distribution(xs)
        fig, ax = plt.subplots()
        ax.hist(results, bins=100, density=True)
        ax.plot(xs, ys)
        plt.show()


if __name__ == '__main__':
    unittest.main()
