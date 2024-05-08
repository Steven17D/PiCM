import unittest

import numpy as np
from matplotlib import pyplot as plt
from acceptance_rejection import get_random_value


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


if __name__ == '__main__':
    unittest.main()
