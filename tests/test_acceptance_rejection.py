import unittest

import numpy as np
from matplotlib import pyplot as plt
from PiCM.probability import get_random_value, maxwell_distribution


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
        v_th = 1
        v_d = 5 * v_th
        min_value, max_value = -v_d * 2, v_d * 2
        distribution = lambda v: maxwell_distribution(v, v_d, v_th)
        results = [get_random_value(distribution, min_value, max_value, v_d) for _ in range(10000)]
        xs = np.linspace(min_value, max_value, 1000)
        ys = distribution(xs)
        fig, ax = plt.subplots()
        ax.hist(results, bins=100, density=True)
        ax.plot(xs, ys)
        plt.show()


if __name__ == '__main__':
    unittest.main()
