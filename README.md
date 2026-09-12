# Simulation of 2D plasma
This code is a Python implementation of the algorithm presented in the paper [Implementation of the two-dimensional electrostatic particlein-cell method](https://doi.org/10.1119/10.0000375).

## Resolution and energy

Set `n` in `PiCM/main.py` to change the grid resolution while keeping the physical
system size `L` fixed. Cell sizes are `delta_r = L / n`. Charge deposition supports
rectangular grids and successive calls with different resolutions.

Field energy is `0.5 * sum(rho * phi) * delta_r[0] * delta_r[1]`. Omitting the
cell area overcounts field energy on finer grids. The periodic Poisson solver
sets the mean potential to zero so an uninitialized Fourier mode cannot
contaminate the field.

Run the numerical regressions with `python -m unittest discover -s tests -p test_simulation.py`.

See the [normalization note](docs/energy-normalization-findings.md) for the
paper references and an isolated analytic counterexample. Run it with
`uv run scripts/check_energy_normalization.py`.

# Example
## Velocity distribution in x axis
![](output/x_streams/20240524-201029-0.png)
![](output/x_streams/20240524-201029-50.png)
![](output/x_streams/20240524-201029-100.png)
![](output/x_streams/20240524-201029-150.png)
![](output/x_streams/20240524-201029-200.png)
![](output/x_streams/20240524-201029-250.png)
![](output/x_streams/20240524-201029-300.png)
![](output/x_streams/20240524-201029-350.png)
![](output/x_streams/20240524-201029-400.png)
![](output/x_streams/20240524-201029-450.png)
![](output/x_streams/20240524-201029-500.png)

## Velocity distribution in x and y axis
![](output/xy_streams/20240524-200257-0.png)
![](output/xy_streams/20240524-200257-50.png)
![](output/xy_streams/20240524-200257-100.png)
![](output/xy_streams/20240524-200257-150.png)
![](output/xy_streams/20240524-200257-200.png)
![](output/xy_streams/20240524-200257-250.png)
![](output/xy_streams/20240524-200257-300.png)
![](output/xy_streams/20240524-200257-350.png)
![](output/xy_streams/20240524-200257-400.png)
![](output/xy_streams/20240524-200257-450.png)
![](output/xy_streams/20240524-200257-500.png)
