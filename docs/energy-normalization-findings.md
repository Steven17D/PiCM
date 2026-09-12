# Electrostatic energy normalization

The mesh stores charge density, not charge per node. Field energy therefore needs the cell area:

```python
field_energy = 0.5 * np.sum(rho * phi) * np.prod(delta_r)
```

For a periodic two-dimensional system with epsilon = 1, integration by parts gives energy per unit out-of-plane length:

$$
U_F=\frac12\int |\nabla\phi|^2\,dx\,dy
   =\frac12\int\rho\phi\,dx\,dy
   \approx\frac{\Delta x\Delta y}{2}\sum_{i,j}\rho_{i,j}\phi_{i,j}.
$$

The old raw sum equals this discrete integral divided by cell area. It rescales only field energy, so adding it to kinetic energy produces an invalid total. If correctly normalized `K + U` were constant, the erroneous total would change by `(1 / cell_area - 1) * change_in_U`.

## Paper and reference code

In [Rodríguez-Patiño et al., American Journal of Physics 88, 159–167, 2020](https://doi.org/10.1119/10.0000375), Eqs. 6 and 8 on page 161 define the mesh quantity as density. Eq. 20 on page 162 omits cell area from field energy. Table II on page 163 uses a 64 by 64 domain and mesh in normalized Debye-length units, so cell area is one and the omission has no numerical effect on that example.

The authors' public implementations contain the same combination:

- C++ revision `15ca11118eaa8400e3b169f87479c1a159723ce4`: [per-area density](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/functions.cc#L83-L110), [raw field energy and ordinary kinetic energy](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/main.cc#L220-L252).
- Python revision `559bfe98a8c67cf72bbd03022a3929051f8a0525`: [per-area density](https://github.com/dfrodriguezp/PiCM/blob/559bfe98a8c67cf72bbd03022a3929051f8a0525/cycle.py#L20-L37), [raw field energy](https://github.com/dfrodriguezp/PiCM/blob/559bfe98a8c67cf72bbd03022a3929051f8a0525/main.py#L245-L284).

This is a diagnostic normalization defect for general grid spacing, not evidence against physical two-stream growth or trapping. The paper's published unit-area example is not invalidated. The reference code was inspected, not executed for this note. No journal erratum is claimed.

## Isolated counterexample

Run `uv run scripts/check_energy_normalization.py`. It samples the same continuum fields on each mesh, without particles, a numerical Poisson solve or time integration:

$$
L_x=L_y=64,\quad k=2\pi/64,\quad
\rho=\cos(kx),\quad\phi=\cos(kx)/k^2,\quad
U_F=L_xL_y/(4k^2).
$$

The exact energy is 106242.96145894798. For 64, 128 and 256 cells per axis, the raw sum divided by exact energy is respectively 1, 4 and 16. The area-weighted sum divided by exact energy is one on all three meshes, to roundoff. The script asserts agreement with the analytic integral at relative tolerance `1e-12`.

These ratios apply to identical sampled fields. Separate particle runs at different resolutions need not produce identical fields.

## Other fixes and limits

This change also makes charge deposition stateless across grid sizes and fixes the uninitialized Poisson DC mode by setting the mean potential to zero. The former deposition's dimensional weights divided by cell area squared were already correctly normalized on square grids; its buffer lifetime and indexing were separate defects. They are not attributed to the paper.

Run the fixture and numerical regressions with `python -m unittest discover -s tests -p test_simulation.py`. They cover charge conservation, periodic rectangular deposition, grid resizing, the discrete Poisson equation, field interpolation, particle updates and correctly normalized energy. Fixture field-energy references are raw sums and are converted by cell area before comparison.

The short multi-resolution regression subsamples the real initial-state fixture for test cost. It is not a full-population trapping demonstration or a convergence study. Exact numerical energy conservation is not guaranteed. Any residual drift requires separate timestep, mesh and particle-count studies before assigning a cause.
