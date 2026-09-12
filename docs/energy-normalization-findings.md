# Missing cell area in the electrostatic energy diagnostic

## Finding

Equation 20 of Rodríguez-Patiño et al., *Implementation of the two-dimensional electrostatic particle-in-cell method*, [American Journal of Physics 88, 159–167, 2020](https://doi.org/10.1119/10.0000375), omits the cell area required when its mesh variable is charge density. Both authors' reference implementations contain the same omission.

This is a diagnostic normalization bug for general grid spacing. The paper's two-stream example uses unit cell area, where the omitted factor is numerically one. This finding does not invalidate that example or identify the physical two-stream instability as a bug. It is a repository technical note, not a published erratum.

## Evidence from the paper

- Equation 1, printed page 160, defines `dx = Lx / nx` and `dy = Ly / ny`.
- Equations 6 and 8, page 161, define the mesh quantity as charge density. The text explicitly calls it "the charge density defined at node (i, j)". A particle's uniform cell density is its charge divided by `dx * dy`.
- Equation 19, page 162, gives kinetic energy as the ordinary mass-weighted particle sum.
- Equation 20, page 162, gives field energy as half the node sum of density times potential, without `dx * dy`. Its explanation of the factor one-half concerns pairwise double counting, not spatial integration.
- Table II, page 163, uses `Lx = Ly = 64`, `nx = ny = 64` in Debye-length units. Thus `dx * dy = 1`. It uses one million particles, `dt = 0.05`, and 1000 steps. Our local comparison is not an exact rerun of those parameters.

The publisher endpoints returned HTTP 403. The inspected copy was a publisher-formatted PDF retrieved from a third-party Sci-Hub mirror, not an authenticated download from AIP. Its cover, DOI, metadata and printed page numbers identify the cited article. SHA-256: `396d3d72ad32285bb499ed620338bc912ed749d11fa84855c83ada6f20522e25`. Equations and definitions above were checked in extracted text. The PDF is not redistributed in this repository.

[CrossRef metadata](https://api.crossref.org/works/10.1119/10.0000375) identifies the article and references the authors' code. The [institutional record](https://hdl.handle.net/11407/5693) provided metadata but no downloadable bitstream. No correction was found in the checked metadata. This is not an exhaustive claim that no correction exists.

## Derivation

Write the mesh density as rho, the potential as phi, and cell area as A = dx * dy. Dimensionless units are fixed while the mesh changes. For a periodic electrostatic system with epsilon = 1,

\[
-\nabla^2\phi=\rho,\qquad \mathbf E=-\nabla\phi.
\]

Integration by parts, with the periodic boundary term cancelling, gives the field energy per unit out-of-plane length:

\[
U_F=\frac12\int |\nabla\phi|^2\,dx\,dy
   =\frac12\int\rho\phi\,dx\,dy
   \approx\frac A2\sum_{i,j}\rho_{i,j}\phi_{i,j}.
\]

If the stored mesh variable were node charge `Q = rho * A`, a raw sum of `Q * phi` would be correct. But the paper and code store density, not node charge. Their dimensional CIC weights sum to A, then divide by A squared; consequently `sum(rho) * A = sum(particle charges)`. The division by A squared in that deposition scheme is not itself a normalization bug.

Define the raw diagnostic `R = 0.5 * sum(rho * phi)`. For the same sampled fields,

\[
R=U_h/A,\qquad U_h:=\frac A2\sum_{i,j}\rho_{i,j}\phi_{i,j}.
\]

On a fixed physical domain, halving both cell widths therefore multiplies the erroneous field-energy scale by four. This identity concerns the diagnostic on given fields. It does not assume that separate particle simulations on different meshes produce identical fields.

The error cannot be repaired by rescaling the plot of total energy. If correctly normalized `K + U_h` were constant while particles exchange energy with the field, then

\[
\Delta(K+R)=(1/A-1)\Delta U_h.
\]

Only the field term has been rescaled. At A = 1/16 the spurious total-energy change is fifteen times the physical field-energy change. An individually normalized field-energy curve can retain its shape, but `K + R` is not physical total energy unless A = 1.

## Reproducible analytic counterexample

Run from the repository root:

```sh
uv run scripts/check_energy_normalization.py
```

The script samples the same exact continuum solution on three meshes, without particles, deposition, a numerical Poisson solve, or time integration:

\[
L_x=L_y=64,\quad k=2\pi/64,\quad
\rho=\cos(kx),\quad \phi=\cos(kx)/k^2.
\]

These fields obey the periodic Poisson equation and have exact energy

\[
U_F=\frac{L_xL_y}{4k^2}=106242.96145894798.
\]

Observed with exit status zero:

- 64 squared cells, A = 1: raw sum / exact energy = 1; integrated / exact = 1.
- 128 squared cells, A = 0.25: raw / exact = 4; integrated / exact = 1.
- 256 squared cells, A = 0.0625: raw / exact = 16; integrated / exact = 1.

Ratios are rounded here. The script prints full floating-point values and asserts the current energy helper agrees with the analytic integral to relative tolerance `1e-12`. This isolates the missing integration factor from particle noise and solver errors.

## Pinned reference implementations

The authors' repositories are distinct from `Steven17D/PiCM`.

C++ revision `15ca11118eaa8400e3b169f87479c1a159723ce4`:

- [main.cc, lines 119–150](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/main.cc#L119-L150): independently configurable spacing and particle masses proportional to physical domain area / particle count.
- [functions.cc, lines 83–110](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/functions.cc#L83-L110): per-area CIC density.
- [functions.cc, lines 164–184](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/functions.cc#L164-L184): spacing-aware Poisson potential, not a potential pre-multiplied by cell area.
- [main.cc, lines 220–252](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/main.cc#L220-L252): normal kinetic energy plus the raw field-energy sum.
- [plot_energy.py, lines 15–28](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/plotters/plot_energy.py#L15-L28): total computed as `KE + FE`, with no downstream correction.
- [build_two_stream.py, lines 6–14](https://github.com/dfrodriguezp/PiCM-cpp/blob/15ca11118eaa8400e3b169f87479c1a159723ce4/build_two_stream.py#L6-L14): unit-area default cells hide the omission.

Python revision `559bfe98a8c67cf72bbd03022a3929051f8a0525` has the same combination: [per-area density in cycle.py](https://github.com/dfrodriguezp/PiCM/blob/559bfe98a8c67cf72bbd03022a3929051f8a0525/cycle.py#L20-L37), [raw field energy in main.py](https://github.com/dfrodriguezp/PiCM/blob/559bfe98a8c67cf72bbd03022a3929051f8a0525/main.py#L245-L284), and [uncorrected total in its plotter](https://github.com/dfrodriguezp/PiCM/blob/559bfe98a8c67cf72bbd03022a3929051f8a0525/plotters/plot_energy.py#L15-L28).

The reference implementations were inspected, not compiled or executed during this investigation. The analytic counterexample executes this repository's corrected helper and evaluates the historical raw expression directly.

## Local simulation evidence and limits

The accepted comparison used baseline `555dce2` and corrected solver `fd720e5`, driven by renderer `957ff7d`. It used the full fixture population of 100,000 particles, including 50,000 movers, a 256 squared mesh on a 64 squared domain, `dt = 0.1`, and 1001 steps. Only scatter markers were thinned, to 5000 plotted movers. The original baseline solver was not replaced with an optimized implementation.

The `picm-faithful-full-comparison` run exited zero. Maximum absolute relative deviation of reported total energy from its initial value was 110.7875 percent before and 0.1337918 percent after. Both results remained finite. The accepted time-30 image shows two-stream trapping in both versions. The movie contains all 16 panels; process-only ffmpeg verification decoded all 1001 frames, 66.73 seconds, at 2400 by 1600 pixels.

The corrected revision also fixes an uninitialized Poisson DC mode and a stale density buffer. Therefore the before/after run is not an area-factor-only experiment. The analytic counterexample above is the isolated evidence for that factor. Those separate Python defects are not attributed to the paper.

The residual 0.1337918 percent has not been decomposed into timestep, spatial-discretization and finite-particle contributions. It is not proof of a particular cause or of convergence. Physical two-stream growth and trapping must be distinguished from both a diagnostic scale error and genuine numerical instability.

The final formalization pass reran the analytic script and the 11 numerical tests in `test_simulation.py`, including its resolution subtests. All passed. This pass changes documentation and adds a proof script; it does not rerun or alter the accepted full-population simulation.

## Recommended next steps

1. Propose the missing-area correction to the authors' reference code, accompanied by the analytic counterexample and pinned source lines. The same report can ask whether Equation 20 merits a clarification or erratum. No issue, pull request or email has been sent; publication is a separate decision.
2. Measure convergence separately from this normalization fix. Keep the physical initial realization and full particle population fixed while scanning mesh size, then timestep at a fixed mesh. Measure correctly normalized total-energy drift and trapping/growth observables. Keep particle-count convergence as a separate scan because particles per cell changes under mesh refinement.
3. Do not generate another comparison movie just to establish the missing factor. The analytic check already isolates it; the accepted local movie preserves the physical behavior.

These are later-work candidates recorded here, not unfinished requirements of this investigation. No upstream patch, convergence claim or journal correction is claimed.

## Cleanup and retained artifacts

The accepted movie and image remain local in `~/Movies/PiCM/full-population-comparison.mp4` and `~/Movies/PiCM/full-population-comparison.png`. Generated media must not be added to Git.

The current branch no longer contains the previously uploaded low-population media. Its branch-based movie URL returned 404. An immutable historical raw URL at removed commit `24b6633` still returned 200; rewriting our branch cannot purge GitHub's retained objects or caches. No claim is made that this historical upload has been erased.

Closeout inventory:

- Local processes: `hub ps` showed all eight PiCM rendering and probe processes exited with status zero. No PiCM process remained to stop. An unrelated shared `omp.browser.headless` process was left untouched; no browser or desktop automation was used.
- Preview environments and cloud baseline: not applicable. This task created no deployment or cloud resource. A scoped repository search found no Dockerfile, Compose, Tilt, Vercel, Fly or Terraform configuration. No unrelated cloud accounts or containers were modified.
- Worktrees and branches: `git worktree list` showed the bare repository and this worktree only. `git branch --merged main` did not include `fix-resolution-energy`. The unmerged source branch and worktree were retained.
- Scratch: removed seven rejected low-population preview files from `~/Movies/PiCM/`, the temporary paper PDF and extracted text, and generated Python/pytest caches. Original repository examples and fixtures were retained.
- Accepted artifacts: directory inspection retained only the final local PNG and MP4. Sizes are 1,530,795 and 154,491,605 bytes respectively. A fresh process-only ffmpeg decode completed all 1001 video frames with exit status zero.
- Deferred work: the upstream correction and separate convergence study are recorded above as later-work candidates. No external issue or message was sent.
- Post-merge health: not applicable; no merge occurred.
- Herdr: not applicable; this task created no pane or session to close.
- Historical upload: remains a remote-retention limitation, described above. Local cleanup and source-only commits do not erase that immutable URL.

An independent adversarial review checked the derivation, paper excerpts, pinned upstream formulas, current helper and counterexample. It found no scientific blockers. A source-path comment was clarified after review. No CI run or fresh full-simulation rerun is claimed for this documentation/proof-script change.
