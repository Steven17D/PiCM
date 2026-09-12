# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy==1.26.2", "tqdm==4.66.4"]
# ///
"""Run with uv run scripts/check_energy_normalization.py.

Sample the same periodic continuum solution on three grids. With epsilon=1,
rho=cos(k*x) and phi=cos(k*x)/k**2 satisfy -laplacian(phi)=rho.
Their exact field energy per unit out-of-plane length is area/(4*k**2).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PiCM.simulation import calculate_field_energy


def main():
    length = np.array([64.0, 64.0])
    k = 2.0 * np.pi / length[0]
    exact = np.prod(length) / (4.0 * k**2)
    results = []
    for cells in (64, 128, 256):
        spacing = length / cells
        x = np.arange(cells) * spacing[0]
        rho = np.broadcast_to(np.cos(k * x)[:, None], (cells, cells))
        phi = rho / k**2
        # Historical PiCM/main.py at 555dce2 omitted the integration cell area.
        raw_sum = 0.5 * np.sum(rho * phi)
        integrated = calculate_field_energy(rho, phi, spacing)
        np.testing.assert_allclose(integrated, exact, rtol=1e-12)
        results.append({
            "cells_per_axis": cells,
            "cell_area": float(np.prod(spacing)),
            "exact_energy": float(exact),
            "raw_sum_over_exact": float(raw_sum / exact),
            "integrated_over_exact": float(integrated / exact),
        })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
