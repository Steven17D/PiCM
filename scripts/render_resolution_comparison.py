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
MP4_PATH = OUTPUT_DIR / "resolution-comparison-long.mp4"
PNG_PATH = OUTPUT_DIR / "resolution-comparison-long.png"

N_CELLS = 256
DT = 0.1
STEPS = 1001
FPS = 15
FIGSIZE = (12.8, 8.0)
DPI = 100
ORANGE = "#e67e22"
BLUE = "#4ea3f1"
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
    _, L, _ = load_config(FIXTURE / "sim_two_stream.json")
    L = np.asarray(L, dtype=np.float64)
    positions, velocities, q_m, moves = (
        values[::50] for values in local_initial_state(FIXTURE / "two_stream.dat")
    )
    n_particles = len(positions)
    if n_particles != 2000:
        raise RuntimeError(f"expected 2000-particle subset, got {n_particles}")
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


def run_version(module, energy_fn, state):
    positions = state["positions"].copy()
    velocities = state["velocities"].copy()
    q_m = state["q_m"].copy()
    charges = state["charges"].copy()
    moves = state["moves"].copy()
    n_moving = int(np.count_nonzero(moves == 1))
    xs = np.empty((STEPS, n_moving), dtype=np.float64)
    ys = np.empty((STEPS, n_moving), dtype=np.float64)
    vxs = np.empty((STEPS, n_moving), dtype=np.float64)
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
        energy[step] = energy_fn(module, vel, rho, phi, state)
    return {"x": xs, "y": ys, "vx": vxs, "energy": energy}


def baseline_energy(module, vel, rho, phi, state):
    kinetic = module.calculate_kinetic_energy(vel, state["mass"])
    return 0.5 * np.sum(rho * phi) + kinetic


def fixed_energy(module, vel, rho, phi, state):
    kinetic = module.calculate_kinetic_energy(vel, state["mass"])
    return module.calculate_field_energy(rho, phi, state["delta_r"]) + kinetic


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


def apply_theme():
    plt.rcParams.update(
        {
            "font.size": 11,
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
        }
    )




def render(old, new, times, L):
    apply_theme()
    old_rel = relative_change(old["energy"])
    new_rel = relative_change(new["energy"])
    old_pct = 100.0 * old_rel
    new_pct = 100.0 * new_rel
    vx_lo, vx_hi = padded_limits(np.concatenate([old["vx"].ravel(), new["vx"].ravel()]))
    e_lo, e_hi = padded_limits(np.concatenate([old_pct, new_pct]))
    e_lo = min(e_lo, -0.05)
    e_hi = max(e_hi, 0.05)
    t_max = times[-1]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.38, "wspace": 0.28},
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.12)
    fig.suptitle(
        "PiCM two-stream  ·  matched inputs  ·  555dce2 vs fd720e5\n"
        "Reported total energy uses E/E_initial − 1 independently per version. "
        "Baseline may drift; nothing was injected.",
        fontsize=13,
    )

    columns = (
        (
            axes[0, 0],
            axes[1, 0],
            old,
            old_pct,
            ORANGE,
            "555dce2  old",
            r"Reported energy: $0.5\sum\rho\phi$ + kinetic (cell area omitted)",
        ),
        (
            axes[0, 1],
            axes[1, 1],
            new,
            new_pct,
            BLUE,
            "fd720e5  fixed",
            r"Reported energy: $0.5\sum\rho\phi\,\Delta x\,\Delta y$ + kinetic",
        ),
    )
    scatters = []
    energy_lines = []
    cursors = []
    drift_texts = []

    for ax_phase, ax_e, run, pct, color, heading, energy_caption in columns:
        ax_phase.set_title(heading, color=color, loc="left", fontsize=12, pad=8)
        ax_phase.set_xlim(0.0, float(L[0]))
        ax_phase.set_ylim(vx_lo, vx_hi)
        ax_phase.set_xlabel(r"$x$")
        ax_phase.set_ylabel(r"$v_x$")
        ax_phase.grid(True)
        scatter = ax_phase.scatter(
            run["x"][0],
            run["vx"][0],
            s=6,
            c=color,
            linewidths=0,
            alpha=0.85,
        )
        scatters.append(scatter)

        ax_e.set_title(energy_caption, fontsize=9, loc="left", pad=6)
        ax_e.set_xlim(0.0, t_max)
        ax_e.set_ylim(e_lo, e_hi)
        ax_e.set_xlabel(r"simulation time  ($\omega_{\mathrm{pe}}t$)")
        ax_e.set_ylabel(r"$(E/E_{\mathrm{initial}}-1)\times 100$  [%]")
        ax_e.axhline(0.0, color="#8b939e", linewidth=0.8, linestyle="--")
        ax_e.grid(True)
        (energy_line,) = ax_e.plot(times[:1], pct[:1], color=color, linewidth=1.8)
        cursor = ax_e.axvline(times[0], color="#d7dde4", linewidth=0.9, alpha=0.85)
        drift = ax_e.text(
            0.02,
            0.92,
            "",
            transform=ax_e.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color=color,
        )
        energy_lines.append(energy_line)
        cursors.append(cursor)
        drift_texts.append(drift)

    fig.text(
        0.5,
        0.025,
        "256×256  ·  dt = 0.1  ·  2000 particles (every 50th of two_stream.dat)  ·  "
        "no injected faults  ·  independent runs of the historical sources",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#b7c0ca",
    )

    def draw(frame):
        for scatter, run in zip(scatters, (old, new)):
            scatter.set_offsets(np.column_stack((run["x"][frame], run["vx"][frame])))
        for line, pct, cursor, drift, series in zip(
            energy_lines,
            (old_pct, new_pct),
            cursors,
            drift_texts,
            (old_rel, new_rel),
        ):
            line.set_data(times[: frame + 1], pct[: frame + 1])
            cursor.set_xdata([times[frame], times[frame]])
            now = series[frame]
            now_txt = "non-finite" if not np.isfinite(now) else f"{100.0 * now:+.3f}%"
            drift.set_text(f"t = {times[frame]:.1f}   drift now {now_txt}")

    plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    writer = FFMpegWriter(
        fps=FPS,
        metadata={"title": "PiCM resolution energy comparison"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(MP4_PATH), DPI):
        for frame in range(STEPS):
            draw(frame)
            writer.grab_frame()
    fig.savefig(PNG_PATH, dpi=DPI)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running matched before/after simulations", flush=True)
    state = load_inputs()
    old_mod = load_simulation(BASELINE, "picm_baseline_555dce2")
    new_mod = load_simulation(FIXED, "picm_fixed_fd720e5")
    old = run_version(old_mod, baseline_energy, state)
    new = run_version(new_mod, fixed_energy, state)
    print("Simulation histories ready; rendering animation", flush=True)
    old_rel = relative_change(old["energy"])
    new_rel = relative_change(new["energy"])
    times = np.arange(STEPS, dtype=np.float64) * DT
    render(old, new, times, state["L"])

    old_finite = bool(np.all(np.isfinite(old["energy"])))
    new_finite = bool(np.all(np.isfinite(new["energy"])))
    old_max = float(np.nanmax(np.abs(old_rel))) if np.any(np.isfinite(old_rel)) else float("nan")
    new_max = float(np.nanmax(np.abs(new_rel))) if np.any(np.isfinite(new_rel)) else float("nan")
    print(f"finite outcome  old={old_finite}  fixed={new_finite}")
    print(f"max abs relative drift  old={old_max:.6e}  fixed={new_max:.6e}")
    print(f"mp4 {MP4_PATH}")
    print(f"png {PNG_PATH}")


if __name__ == "__main__":
    main()
