"""
Based on chapter 3.3 "The acceptance-rejection method" from Statistical Data Analysis
"""
import numpy as np


def maxwell_distribution(v, v_d, v_th):
    """
    Velocity distribution
    """
    n_e = 0.5  # In order to normalize area
    return (
            (n_e / np.sqrt(2 * np.pi * (v_th ** 2))) * (
            np.exp((-(v - v_d) ** 2) / (2 * (v_th ** 2))) + np.exp((-(v + v_d) ** 2) / (2 * (v_th ** 2)))
    )
    )


def get_random_value(distribution_function, x_min, x_max, f_max):
    """
    Function to generate a random value from a distribution function
    """
    while True:
        x = np.random.uniform(x_min, x_max)
        u = np.random.uniform(0, f_max)
        f_x = distribution_function(x)
        if u < f_x:
            return x
