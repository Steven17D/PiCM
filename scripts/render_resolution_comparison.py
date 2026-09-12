# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "numpy==1.26.2",
#     "matplotlib==3.8.2",
#     "tqdm==4.66.4",
#     "imageio-ffmpeg==0.6.0",
# ]
# ///
"""Render with: uv run scripts/render_resolution_comparison.py."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PiCM.loader import load_config, local_initial_state

BASELINE = "555dce2"
FIXED = "fd720e5"
FIXTURE = REPO / "tests" / "electrosctatic"
OUTPUT_DIR = Path.home() / "Movies" / "PiCM"
MP4_PATH = OUTPUT_DIR / "full-population-comparison.mp4"
PNG_PATH = OUTPUT_DIR / "full-population-comparison.png"

N_CELLS = 256
DT = 0.1
STEPS = 1001
FPS = 15
FIGSIZE = (24.0, 16.0)
DPI = 100
HIST_BINS = 50
PREVIEW_FRAME = 300
# Subsampling the simulation changes noise and particle trapping.
# Keep every particle in the physics; thin only the scatter drawings.
SCATTER_STRIDE = 10
ORANGE = "#e67e22"
BLUE = "#4ea3f1"
RED = "#e05d5d"
KINETIC_COLOR = "#f0c14b"
FIELD_COLOR = "#7fd99a"
TOTAL_COLOR = "#f2f5f8"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load_simulation(commit: str, name: str):
    source = subprocess.check_output(
        ["git", "show", f"{commit}:PiCM/simulation.py"],
        cwd=REPO,
    )
    module = types.ModuleType(name)
    module.__file__ = f"<{commit}:PiCM/simulation.py>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    real_tqdm = module.tqdm

    def tqdm_disabled(iterable, *args, **kwargs):
        kwargs["disable"] = True
        return real_tqdm(iterable, *args, **kwargs)

    module.tqdm = tqdm_disabled
    return module


def load_inputs():
    n_expected, L, _ = load_config(FIXTURE / "sim_two_stream.json")
    L = np.asarray(L, dtype=np.float64)
    positions, velocities, q_m, moves = local_initial_state(FIXTURE / "two_stream.dat")
    n_particles = len(positions)
    if n_particles != n_expected:
        raise RuntimeError(
            f"expected {n_expected} particles from load_config, got {n_particles}"
        )
    charges = np.prod(L) * q_m / n_particles
    mass = np.prod(L) / n_particles
    n = np.array([N_CELLS, N_CELLS], dtype=int)
    delta_r = L / n
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "velocities": np.asarray(velocities, dtype=np.float64),
        "q_m": np.asarray(q_m, dtype=np.float64),
        "charges": np.asarray(charges, dtype=np.float64),
        "moves": np.asarray(moves, dtype=np.float64),
        "mass": float(mass),
        "L": L,
        "n": n,
        "delta_r": delta_r,
        "B": np.zeros(3, dtype=np.float64),
    }


def baseline_field_energy(_module, rho, phi, _state):
    return 0.5 * float(np.dot(rho.ravel(), phi.ravel()))


def fixed_field_energy(module, rho, phi, state):
    return float(module.calculate_field_energy(rho, phi, state["delta_r"]))


def periodic_x(x, lx):
    return np.mod(x + 0.5 * lx, lx) - 0.5 * lx


def rolled_xy_image(field):
    return np.roll(field, field.shape[0] // 2, axis=0).T


def run_version(module, field_energy_fn, state, label):
    print(f"run {label}", flush=True)
    positions = state["positions"].copy()
    velocities = state["velocities"].copy()
    q_m = state["q_m"].copy()
    charges = state["charges"].copy()
    moves = state["moves"].copy()
    n_moving = int(np.count_nonzero(moves == 1))
    nx, ny = int(state["n"][0]), int(state["n"][1])
    xs = np.empty((STEPS, n_moving), dtype=np.float32)
    ys = np.empty((STEPS, n_moving), dtype=np.float32)
    vxs = np.empty((STEPS, n_moving), dtype=np.float32)
    vys = np.empty((STEPS, n_moving), dtype=np.float32)
    rhos = np.empty((STEPS, nx, ny), dtype=np.float32)
    phis = np.empty((STEPS, nx, ny), dtype=np.float32)
    kinetic = np.empty(STEPS, dtype=np.float64)
    field = np.empty(STEPS, dtype=np.float64)
    energy = np.empty(STEPS, dtype=np.float64)
    for moving_xy, vel, rho, phi, _, step in module.simulate(
        positions,
        velocities,
        q_m,
        charges,
        moves,
        state["L"],
        state["n"],
        state["delta_r"],
        state["B"],
        DT,
        STEPS,
    ):
        xs[step] = moving_xy[:, 0]
        ys[step] = moving_xy[:, 1]
        vxs[step] = vel[:, 0]
        vys[step] = vel[:, 1]
        rhos[step] = rho
        phis[step] = phi
        kinetic[step] = module.calculate_kinetic_energy(vel, state["mass"])
        field[step] = field_energy_fn(module, rho, phi, state)
        energy[step] = kinetic[step] + field[step]
        if step % 250 == 0 or step + 1 == STEPS:
            print(f"run {label}  step {step}/{STEPS - 1}", flush=True)
    print(f"run {label} done", flush=True)
    return {
        "x": xs,
        "y": ys,
        "vx": vxs,
        "vy": vys,
        "rho": rhos,
        "phi": phis,
        "kinetic": kinetic,
        "field": field,
        "energy": energy,
    }


def relative_change(energy):
    initial = energy[0]
    if not np.isfinite(initial) or initial == 0.0:
        return np.full_like(energy, np.nan)
    return energy / initial - 1.0


def padded_limits(values, pad=0.08):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(finite.min())
    hi = float(finite.max())
    span = hi - lo
    if span == 0.0:
        span = max(abs(hi), 1.0)
    return lo - pad * span, hi + pad * span


def shared_clim(*fields):
    lo = min(float(field.min()) for field in fields)
    hi = max(float(field.max()) for field in fields)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if lo == hi:
        span = max(abs(hi), 1.0)
        return lo - 0.05 * span, hi + 0.05 * span
    return lo, hi


def histogram_series(values, bin_edges):
    counts = np.empty((values.shape[0], bin_edges.size - 1), dtype=np.float64)
    for frame, sample in enumerate(values):
        counts[frame], _ = np.histogram(sample, bins=bin_edges, density=True)
    return counts


def energy_limits(kinetic, field, total):
    vals = np.concatenate([kinetic, field, total])
    lo, hi = padded_limits(vals)
    finite = vals[np.isfinite(vals)]
    if finite.size and float(finite.min()) >= 0.0:
        lo = 0.0
        hi = max(hi, float(finite.max()) * 1.1)
    return lo, hi


def apply_theme():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "#16181b",
            "axes.facecolor": "#1f2328",
            "axes.edgecolor": "#8b939e",
            "axes.labelcolor": "#e8edf2",
            "text.color": "#e8edf2",
            "xtick.color": "#c6cdd6",
            "ytick.color": "#c6cdd6",
            "grid.color": "#3a414a",
            "grid.alpha": 0.85,
            "savefig.facecolor": "#16181b",
            "legend.facecolor": "#1f2328",
            "legend.edgecolor": "#8b939e",
        }
    )


def add_dashboard(
    ax_top,
    ax_bot,
    run,
    *,
    times,
    L,
    v_lim,
    bin_edges,
    hist_ylim,
    hist_vx,
    hist_vy,
    phi_clim,
    rho_clim,
    energy_lim,
    accent,
    heading,
    field_caption,
    particle_colors,
):
    ax_vx, ax_vy, ax_phi, ax_xy = ax_top
    ax_vx_h, ax_vy_h, ax_rho, ax_energy = ax_bot
    lx, ly = float(L[0]), float(L[1])
    extent = (-0.5 * lx, 0.5 * lx, 0.0, ly)
    t_max = float(times[-1])
    v_lo, v_hi = v_lim
    dots = slice(None, None, SCATTER_STRIDE)
    x_dots = periodic_x(run["x"][0][dots], lx)
    y_dots = run["y"][0][dots]
    vx_dots = run["vx"][0][dots]
    vy_dots = run["vy"][0][dots]

    ax_vx.set_title(f"{heading}   " + r"$x$-$v_x$", color=accent, loc="left", pad=8)
    ax_vx.set_xlim(-0.5 * lx, 0.5 * lx)
    ax_vx.set_ylim(v_lo, v_hi)
    ax_vx.set_xlabel(r"$x / \lambda_D$")
    ax_vx.set_ylabel(r"$v_x / v_{\rm th}$")
    ax_vx.grid(True)
    vx_scatter = ax_vx.scatter(
        x_dots,
        vx_dots,
        s=5,
        c=particle_colors,
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )

    ax_vy.set_title(r"$y$-$v_y$", loc="left", pad=8)
    ax_vy.set_xlim(0.0, ly)
    ax_vy.set_ylim(v_lo, v_hi)
    ax_vy.set_xlabel(r"$y / \lambda_D$")
    ax_vy.set_ylabel(r"$v_y / v_{\rm th}$")
    ax_vy.grid(True)
    vy_scatter = ax_vy.scatter(
        y_dots,
        vy_dots,
        s=5,
        c=particle_colors,
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )

    ax_phi.set_title(r"potential  $\phi$", loc="left", pad=8)
    ax_phi.set_xlabel(r"$x / \lambda_D$")
    ax_phi.set_ylabel(r"$y / \lambda_D$")
    phi_im = ax_phi.imshow(
        rolled_xy_image(run["phi"][0]),
        origin="lower",
        extent=extent,
        cmap="jet",
        interpolation="nearest",
        aspect="auto",
        vmin=phi_clim[0],
        vmax=phi_clim[1],
        rasterized=True,
    )
    phi_bar = plt.colorbar(phi_im, ax=ax_phi, fraction=0.046, pad=0.02)
    phi_bar.set_label(r"$\phi / (T_e / e)$")

    ax_xy.set_title(r"$x$-$y$ positions", loc="left", pad=8)
    ax_xy.set_xlim(-0.5 * lx, 0.5 * lx)
    ax_xy.set_ylim(0.0, ly)
    ax_xy.set_xlabel(r"$x / \lambda_D$")
    ax_xy.set_ylabel(r"$y / \lambda_D$")
    ax_xy.set_aspect("auto")
    ax_xy.grid(True)
    xy_scatter = ax_xy.scatter(
        x_dots,
        y_dots,
        s=5,
        c=particle_colors,
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )

    ax_vx_h.set_title(r"$v_x$ histogram", loc="left", pad=8)
    ax_vx_h.set_xlim(bin_edges[0], bin_edges[-1])
    ax_vx_h.set_ylim(0.0, hist_ylim)
    ax_vx_h.set_xlabel(r"$v_x / v_{\rm th}$")
    ax_vx_h.set_ylabel("density")
    ax_vx_h.grid(True)
    _, _, vx_bars = ax_vx_h.hist(
        run["vx"][0],
        bins=bin_edges,
        density=True,
        color=RED,
        edgecolor="none",
    )

    ax_vy_h.set_title(r"$v_y$ histogram", loc="left", pad=8)
    ax_vy_h.set_xlim(bin_edges[0], bin_edges[-1])
    ax_vy_h.set_ylim(0.0, hist_ylim)
    ax_vy_h.set_xlabel(r"$v_y / v_{\rm th}$")
    ax_vy_h.set_ylabel("density")
    ax_vy_h.grid(True)
    _, _, vy_bars = ax_vy_h.hist(
        run["vy"][0],
        bins=bin_edges,
        density=True,
        color=RED,
        edgecolor="none",
    )

    ax_rho.set_title(r"charge density  $\rho$", loc="left", pad=8)
    ax_rho.set_xlabel(r"$x / \lambda_D$")
    ax_rho.set_ylabel(r"$y / \lambda_D$")
    rho_im = ax_rho.imshow(
        rolled_xy_image(run["rho"][0]),
        origin="lower",
        extent=extent,
        cmap="jet",
        interpolation="nearest",
        aspect="auto",
        vmin=rho_clim[0],
        vmax=rho_clim[1],
        rasterized=True,
    )
    rho_bar = plt.colorbar(rho_im, ax=ax_rho, fraction=0.046, pad=0.02)
    rho_bar.set_label(r"$\rho / (e\lambda_D^{-2})$")

    ax_energy.set_title(field_caption, loc="left", fontsize=10, pad=8)
    ax_energy.set_xlim(0.0, t_max)
    ax_energy.set_ylim(*energy_lim)
    ax_energy.set_xlabel(r"$\omega_{\rm pe}t$")
    ax_energy.set_ylabel(r"$E / (n_0 T_e / \varepsilon_0)$")
    ax_energy.grid(True)
    (kinetic_line,) = ax_energy.plot(
        times[:1], run["kinetic"][:1], color=KINETIC_COLOR, linewidth=1.6, label="Kinetic"
    )
    (field_line,) = ax_energy.plot(
        times[:1], run["field"][:1], color=FIELD_COLOR, linewidth=1.6, label="Field"
    )
    (total_line,) = ax_energy.plot(
        times[:1], run["energy"][:1], color=TOTAL_COLOR, linewidth=1.8, label="Total"
    )
    cursor = ax_energy.axvline(times[0], color="#d7dde4", linewidth=0.9, alpha=0.85)
    ax_energy.legend(loc="upper right", framealpha=0.92)
    drift = ax_energy.text(
        0.02,
        0.96,
        "",
        transform=ax_energy.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color=accent,
        bbox={"facecolor": plt.rcParams["axes.facecolor"], "edgecolor": "none"},
    )
    return {
        "vx_scatter": vx_scatter,
        "vy_scatter": vy_scatter,
        "xy_scatter": xy_scatter,
        "phi_im": phi_im,
        "rho_im": rho_im,
        "vx_bars": vx_bars,
        "vy_bars": vy_bars,
        "hist_vx": hist_vx,
        "hist_vy": hist_vy,
        "kinetic_line": kinetic_line,
        "field_line": field_line,
        "total_line": total_line,
        "cursor": cursor,
        "drift": drift,
        "run": run,
    }


def set_bar_heights(container, counts):
    for rect, count in zip(container.patches, counts):
        rect.set_height(count)


def format_drift(value):
    if not np.isfinite(value):
        return "non-finite"
    return f"{100.0 * value:+.3f}%"


def render(old, new, times, L, n_particles):
    apply_theme()
    old_rel = relative_change(old["energy"])
    new_rel = relative_change(new["energy"])
    v_lim = padded_limits(
        np.concatenate(
            [old["vx"].ravel(), new["vx"].ravel(), old["vy"].ravel(), new["vy"].ravel()]
        )
    )
    bin_edges = np.linspace(v_lim[0], v_lim[1], HIST_BINS + 1)
    old_hist_vx = histogram_series(old["vx"], bin_edges)
    new_hist_vx = histogram_series(new["vx"], bin_edges)
    old_hist_vy = histogram_series(old["vy"], bin_edges)
    new_hist_vy = histogram_series(new["vy"], bin_edges)
    hist_peak = max(
        float(old_hist_vx.max()),
        float(new_hist_vx.max()),
        float(old_hist_vy.max()),
        float(new_hist_vy.max()),
        0.20,
    )
    hist_ylim = hist_peak * 1.1
    phi_clim = shared_clim(old["phi"], new["phi"])
    rho_clim = shared_clim(old["rho"], new["rho"])
    old_e_lim = energy_limits(old["kinetic"], old["field"], old["energy"])
    new_e_lim = energy_limits(new["kinetic"], new["field"], new["energy"])
    shared_e_lim = (min(old_e_lim[0], new_e_lim[0]), max(old_e_lim[1], new_e_lim[1]))
    dots = slice(None, None, SCATTER_STRIDE)
    old_colors = np.where(old["vx"][0][dots] < 0.0, BLUE, RED)
    new_colors = np.where(new["vx"][0][dots] < 0.0, BLUE, RED)
    n_moving = old["x"].shape[1]
    n_dots = n_moving // SCATTER_STRIDE
    lx = float(L[0])

    fig, axes = plt.subplots(4, 4, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.065, hspace=0.42, wspace=0.30)
    title = fig.suptitle("", fontsize=15, y=0.975)
    fig.text(
        0.018,
        0.70,
        f"BEFORE\n{BASELINE}",
        rotation=90,
        va="center",
        ha="center",
        color=ORANGE,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.018,
        0.28,
        f"AFTER\n{FIXED}",
        rotation=90,
        va="center",
        ha="center",
        color=BLUE,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        f"256x256  ·  dt = 0.1  ·  {STEPS} frames  ·  {n_particles} simulated particles  ·  "
        f"{n_dots} moving dots (stride {SCATTER_STRIDE})  ·  "
        r"periodic $x$ window $[-L_x/2, L_x/2)$  ·  "
        "no injected faults  ·  independent historical sources 555dce2 / fd720e5  ·  "
        "normalized total drift = E/E_initial - 1 per version",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#b7c0ca",
    )

    dashboards = (
        add_dashboard(
            axes[0],
            axes[1],
            old,
            times=times,
            L=L,
            v_lim=v_lim,
            bin_edges=bin_edges,
            hist_ylim=hist_ylim,
            hist_vx=old_hist_vx,
            hist_vy=old_hist_vy,
            phi_clim=phi_clim,
            rho_clim=rho_clim,
            energy_lim=shared_e_lim,
            accent=ORANGE,
            heading=f"BEFORE  {BASELINE}",
            field_caption="Field 0.5 sum(rho phi)  (cell area omitted)  + kinetic",
            particle_colors=old_colors,
        ),
        add_dashboard(
            axes[2],
            axes[3],
            new,
            times=times,
            L=L,
            v_lim=v_lim,
            bin_edges=bin_edges,
            hist_ylim=hist_ylim,
            hist_vx=new_hist_vx,
            hist_vy=new_hist_vy,
            phi_clim=phi_clim,
            rho_clim=rho_clim,
            energy_lim=shared_e_lim,
            accent=BLUE,
            heading=f"AFTER  {FIXED}",
            field_caption="Field 0.5 sum(rho phi) dx dy  + kinetic",
            particle_colors=new_colors,
        ),
    )
    drifts = (old_rel, new_rel)

    def draw(frame):
        title.set_text(
            "PiCM two-stream  ·  matched inputs  ·  "
            f"{BASELINE} vs {FIXED}\n"
            rf"$\omega_{{\rm pe}}t = {times[frame]:.2f}$"
            f"  ·  {n_particles} simulated particles  ·  {n_dots} moving dots"
            r"  ·  periodic $x$ window $[-L_x/2, L_x/2)$"
            f"  ·  {STEPS} frames  ·  all 8 original diagrams per version"
        )
        for artists, rel in zip(dashboards, drifts):
            run = artists["run"]
            x_dots = periodic_x(run["x"][frame][dots], lx)
            y_dots = run["y"][frame][dots]
            vx_dots = run["vx"][frame][dots]
            vy_dots = run["vy"][frame][dots]
            artists["vx_scatter"].set_offsets(np.column_stack((x_dots, vx_dots)))
            artists["vy_scatter"].set_offsets(np.column_stack((y_dots, vy_dots)))
            artists["xy_scatter"].set_offsets(np.column_stack((x_dots, y_dots)))
            artists["phi_im"].set_data(rolled_xy_image(run["phi"][frame]))
            artists["rho_im"].set_data(rolled_xy_image(run["rho"][frame]))
            set_bar_heights(artists["vx_bars"], artists["hist_vx"][frame])
            set_bar_heights(artists["vy_bars"], artists["hist_vy"][frame])
            artists["kinetic_line"].set_data(times[: frame + 1], run["kinetic"][: frame + 1])
            artists["field_line"].set_data(times[: frame + 1], run["field"][: frame + 1])
            artists["total_line"].set_data(times[: frame + 1], run["energy"][: frame + 1])
            artists["cursor"].set_xdata([times[frame], times[frame]])
            now = rel[frame]
            peak = float(np.nanmax(np.abs(rel))) if np.any(np.isfinite(rel)) else float("nan")
            peak_txt = "non-finite" if not np.isfinite(peak) else f"{100.0 * peak:.3f}%"
            artists["drift"].set_text(
                f"Total drift {format_drift(now)}\n"
                f"Peak |drift| {peak_txt}"
            )

    plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    writer = FFMpegWriter(
        fps=FPS,
        metadata={"title": "PiCM full-population comparison"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    print("render start", flush=True)
    with writer.saving(fig, str(MP4_PATH), DPI):
        for frame in range(STEPS):
            draw(frame)
            writer.grab_frame()
            if frame % 100 == 0 or frame + 1 == STEPS:
                print(f"render {frame}/{STEPS - 1}", flush=True)
    draw(PREVIEW_FRAME)
    fig.savefig(PNG_PATH, dpi=DPI)
    plt.close(fig)
    print("render done", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_inputs()
    n_particles = len(state["positions"])
    print(
        f"start  n={N_CELLS}  dt={DT}  steps={STEPS}  fps={FPS}  "
        f"N={n_particles}  out={MP4_PATH}",
        flush=True,
    )
    old_mod = load_simulation(BASELINE, "picm_baseline_555dce2")
    new_mod = load_simulation(FIXED, "picm_fixed_fd720e5")
    old = run_version(old_mod, baseline_field_energy, state, f"before {BASELINE}")
    new = run_version(new_mod, fixed_field_energy, state, f"after {FIXED}")
    times = np.arange(STEPS, dtype=np.float64) * DT
    render(old, new, times, state["L"], n_particles)

    old_rel = relative_change(old["energy"])
    new_rel = relative_change(new["energy"])
    old_finite = bool(
        np.all(np.isfinite(old["kinetic"]))
        and np.all(np.isfinite(old["field"]))
        and np.all(np.isfinite(old["energy"]))
    )
    new_finite = bool(
        np.all(np.isfinite(new["kinetic"]))
        and np.all(np.isfinite(new["field"]))
        and np.all(np.isfinite(new["energy"]))
    )
    old_max = float(np.nanmax(np.abs(old_rel))) if np.any(np.isfinite(old_rel)) else float("nan")
    new_max = float(np.nanmax(np.abs(new_rel))) if np.any(np.isfinite(new_rel)) else float("nan")
    print(f"finite outcome  old={old_finite}  fixed={new_finite}")
    print(f"max abs relative drift  old={old_max:.6e}  fixed={new_max:.6e}")
    print(f"mp4 {MP4_PATH}")
    print(f"png {PNG_PATH}")


if __name__ == "__main__":
    main()
