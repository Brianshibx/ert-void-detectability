#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project A -- Detectability envelope of a water-filled cavity with 2D ERT
=======================================================================

Idea
----
Treat (forward modelling + inversion) as a black box.  Put a known
water-filled cavity into a synthetic half-space, run the full pipeline,
and ask: *can the inversion get it back?*  Repeat over size, depth,
resistivity contrast and noise.  Decide "detected" with a threshold
calibrated on noise-only inversions, so the answer is statistical rather
than "I think I can see an anomaly".

Usage
-----
    python detectability.py --smoke                # 1 case, make sure it works
    python detectability.py --null 50 --null-only  # calibrate threshold
    python detectability.py --fast                 # 6 cases, check timing
    python detectability.py --full --jobs 4        # full sweep (675 cases)
    python detectability.py --plot                 # figures
    python detectability.py --examples             # success/failure panels

Notes
-----
* The forward mesh uses `area_fwd`; the inversion uses ERTManager's own
  mesh.  This mismatch is deliberate: it avoids the "inverse crime".
* `lam` is fixed for every run.  Changing it mid-sweep compares
  different amounts of smoothing, not different detectability.
* 2D means an infinitely long cylinder -> optimistic.  State this as a
  limitation in any write-up.

Written as a starting skeleton.  pyGIMLi APIs drift between versions;
if a call fails, see README section 9.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import traceback
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pygimli as pg
import pygimli.physics.ert as ert
import pygimli.meshtools as mt

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except Exception:                                    # pragma: no cover
    HAVE_JOBLIB = False


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass
class Cfg:
    n_elec: int = 25                 # 25 electrodes
    spacing: float = 2.0             # 2 m -> 48 m spread
    x_start: float = 0.0
    lam: float = 20.0                # FIXED for the whole study
    noise_abs: float = 1e-5
    host: float = 300.0              # ohm.m
    cover_t: float = 0.0             # conductive overburden thickness (0 = off)
    cover_rho: float = 20.0
    area_fwd: float = 1.0            # max cell area of the FORWARD mesh
    inv_max_cell: float = 2.0        # max cell size of the INVERSION mesh
    x_min: float = -40.0
    x_max: float = 88.0
    z_bot: float = -60.0


# sweep grid (edit freely)
SIZES = [1.5, 2.0, 3.0, 4.0, 6.0]    # cavity radius, m
TOPS = [5.0, 10.0, 15.0, 22.0, 30.0]  # depth of the cavity TOP, m
RHOS = [1.0, 5.0, 30.0]              # cavity resistivity, ohm.m
NOISES = [0.01, 0.02, 0.05]          # relative noise
SEEDS = list(range(1, 11))


# ----------------------------------------------------------------------
# geometry / forward setup
# ----------------------------------------------------------------------
def electrodes(cfg: Cfg) -> np.ndarray:
    return cfg.x_start + np.arange(cfg.n_elec) * cfg.spacing


def make_scheme(cfg: Cfg):
    """Wenner-alpha.  Swap 'wa' for 'dd' to try dipole-dipole."""
    elec = electrodes(cfg)
    fn = getattr(ert, "createData", None) or getattr(ert, "createERTData", None)
    if fn is None:
        raise RuntimeError("this pyGIMLi has neither createData nor createERTData")
    return fn(elecs=elec, schemeName="wa")


def quiet():
    """Turn off pyGIMLi chatter -- the name of this function has changed
    between versions, so try the known spellings and ignore failures."""
    fn = getattr(pg, "setVerbosityLevel", None)
    if fn is None and hasattr(pg, "core"):
        fn = getattr(pg.core, "setVerbosityLevel", None)
    if callable(fn):
        try:
            fn(0)
        except Exception:
            pass
    try:
        import logging
        logging.disable(logging.INFO)          # global: kill INFO chatter
        for name in ("pygimli", "pyGIMLi", "pygimli.core"):
            logging.getLogger(name).setLevel(logging.WARNING)
    except Exception:
        pass


def build_mesh(cfg: Cfg, xc=None, zc=None, r=None):
    """PLC -> mesh.  If xc is given, a circular cavity marker=2 is added."""
    if cfg.cover_t > 0:
        top = mt.createRectangle(start=[cfg.x_min, 0.0],
                                 end=[cfg.x_max, -cfg.cover_t], marker=3)
        bot = mt.createRectangle(start=[cfg.x_min, -cfg.cover_t],
                                 end=[cfg.x_max, cfg.z_bot], marker=1)
        plc = mt.mergePLC([top, bot])
    else:
        plc = mt.createWorld(start=[cfg.x_min, 0.0],
                             end=[cfg.x_max, cfg.z_bot], marker=1)

    if xc is not None:
        cavity = mt.createCircle(pos=[xc, zc], radius=r, marker=2)
        plc = mt.mergePLC([plc, cavity])

    kw = dict(quality=34)
    try:
        return mt.createMesh(plc, area=cfg.area_fwd, **kw)
    except TypeError:                      # older pyGIMLi: no `area`
        return mt.createMesh(plc, **kw)


def rho_map(cfg: Cfg, mesh, rho_cav=None):
    pairs = [[1, cfg.host]]
    if cfg.cover_t > 0:
        pairs.append([3, cfg.cover_rho])
    if rho_cav is not None:
        pairs.append([2, float(rho_cav)])
    fn = getattr(pg.solver, "parseMapToCellArray", None)
    if fn is None and hasattr(pg, "core"):
        fn = getattr(pg.core, "parseMapToCellArray", None)
    return fn(pairs, mesh)


# ----------------------------------------------------------------------
# one forward + inverse
# ----------------------------------------------------------------------
def build_inversion_mesh(cfg: Cfg):
    """A para-mesh fine enough to resolve a few-metre cavity.

    Guarded: if this pyGIMLi has no createParaMesh, return None and let
    ERTManager build its own (coarser) mesh.
    """
    fn = getattr(mt, "createParaMesh", None)
    if fn is None:
        return None
    try:
        return fn(electrodes(cfg), paraDepth=35.0, paraDX=0.5,
                  paraMaxCellSize=cfg.inv_max_cell,
                  boundaryMaxCellSize=25.0)
    except Exception as e:
        print(f"  [mesh] createParaMesh unavailable ({e})")
        return None


def simulate_and_invert(cfg: Cfg, mesh, rho_cav, noise, seed, scheme):
    res = rho_map(cfg, mesh, rho_cav)

    try:
        data = ert.simulate(mesh, scheme=scheme, res=res,
                            noiseLevel=noise, noiseAbs=cfg.noise_abs,
                            seed=seed)
    except TypeError:                       # seed not supported in this build
        data = ert.simulate(mesh, scheme=scheme, res=res,
                            noiseLevel=noise, noiseAbs=cfg.noise_abs)

    # make sure geometric factors are present (version dependent)
    try:
        _ = data["k"]
    except Exception:
        data["k"] = ert.geometricFactors(data)

    mgr = ert.ERTManager(data, verbose=False)

    inv_mesh = build_inversion_mesh(cfg)
    if inv_mesh is not None and hasattr(mgr, "setMesh"):
        try:
            mgr.setMesh(inv_mesh)
        except Exception as e:
            print(f"  [mesh] setMesh failed ({e})")

    try:
        mgr.invert(lam=cfg.lam, verbose=False)
    except TypeError:                       # older/newer signature
        mgr.invert(lam=cfg.lam)

    model = np.asarray(mgr.model, dtype=float)

    mesh_inv = getattr(mgr, "paraDomain", None)
    if mesh_inv is None:
        mesh_inv = mgr.mesh
    cells = np.asarray(mesh_inv.cellCenters(), dtype=float)
    return model, cells, mesh_inv


def _sample(cells, vals, x, z, k=8):
    """Inverse-distance weighted mean of the k nearest cells.

    Mesh-independent, so it still works when the inversion mesh is coarse
    and a small cavity contains fewer than one cell.
    """
    d = np.hypot(cells[:, 0] - x, cells[:, 1] - z)
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 0.5)
    return float(np.sum(vals[idx] * w) / np.sum(w))


def amp_metric(model, cells, xc, zc, r):
    """Recovered contrast between a ring around the cavity and its centre.

    Positive  -> the centre is more conductive than its surroundings,
                 which is what a water-filled cavity should look like.
    """
    m = np.log10(np.clip(model, 1e-3, None))

    centre = _sample(cells, m, xc, zc)

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring = np.mean([_sample(cells, m, xc + 2.5 * r * np.cos(a),
                            zc + 2.5 * r * np.sin(a)) for a in angles])

    amp = ring - centre

    # localisation error: distance from the true centre to the most
    # conductive cell within 2.5 r
    d = np.hypot(cells[:, 0] - xc, cells[:, 1] - zc)
    near = d <= 2.5 * r
    if near.sum() > 0:
        masked = np.where(near, m, np.inf)
        loc = float(d[int(np.argmin(masked))])
    else:
        loc = float("nan")
    return amp, loc


# ----------------------------------------------------------------------
# null distribution: noise only, no cavity
# ----------------------------------------------------------------------
def _null_task(cfg_dict, noise, k, sizes, tops):
    """One noise-only inversion + metric evaluated at every region.

    `cfg` travels as a plain dict: a dataclass defined in __main__
    cannot be pickled to a joblib worker.
    """
    cfg = Cfg(**cfg_dict)
    mesh = build_mesh(cfg)                       # no cavity
    scheme = make_scheme(cfg)
    model, cells, _ = simulate_and_invert(cfg, mesh, None, noise,
                                          1000 + k, scheme)
    xc = cfg.x_start + (cfg.n_elec - 1) * cfg.spacing / 2.0
    out = []
    for r in sizes:
        for top in tops:
            amp, _ = amp_metric(model, cells, xc, -(top + r), r)
            out.append((noise, r, top, amp))
    return out


def make_null_table(cfg: Cfg, n: int, scheme, sizes, tops, noises, jobs=1):
    """Invert homogeneous data n times per noise level.

    For every null inversion we evaluate the *same* metric at every
    (size, top) region.  Cheap, and gives a per-combination threshold.
    """
    cfg_dict = asdict(cfg)
    tasks = [(cfg_dict, noise, k, sizes, tops)
             for noise in noises for k in range(n)]
    total = len(tasks)
    print(f"  {total} null inversions; the first one takes the longest "
          f"(mesh build).  Ctrl+C to stop.", flush=True)

    t0 = time.time()
    if jobs > 1 and HAVE_JOBLIB:
        chunks = Parallel(n_jobs=jobs, verbose=10)(
            delayed(_null_task)(*t) for t in tasks)
    else:
        chunks = []
        for i, t in enumerate(tasks, 1):
            chunks.append(_null_task(*t))
            el = time.time() - t0
            eta = el / i * (total - i)
            print(f"  null {i}/{total}  elapsed {el / 60:.1f} min  "
                  f"eta {eta / 60:.1f} min  ({el / i:.1f} s each)",
                  flush=True)

    rows = [r for sub in chunks for r in sub]
    return pd.DataFrame(rows, columns=["noise", "size", "top", "amp"])


def thresholds_from_null(null_df: pd.DataFrame) -> pd.DataFrame:
    g = null_df.groupby(["noise", "size", "top"])["amp"]
    out = g.quantile(0.95).reset_index().rename(columns={"amp": "thr"})
    return out


# ----------------------------------------------------------------------
# one sweep case
# ----------------------------------------------------------------------
def run_case(cfg_dict, size, top, rho_cav, noise, seed):
    cfg = Cfg(**cfg_dict)
    t0 = time.time()
    xc = cfg.x_start + (cfg.n_elec - 1) * cfg.spacing / 2.0
    zc = -(top + size)

    scheme = make_scheme(cfg)
    mesh = build_mesh(cfg, xc=xc, zc=zc, r=size)
    model, cells, _ = simulate_and_invert(cfg, mesh, rho_cav, noise,
                                          seed, scheme)
    amp, loc = amp_metric(model, cells, xc, zc, size)

    return dict(size=size, top=top, rho_cav=rho_cav, noise=noise,
                seed=seed, amp=amp, loc_err=loc,
                t_sec=round(time.time() - t0, 2))


def safe_run_case(*args):
    """joblib worker: never let one bad case kill the whole sweep."""
    try:
        return run_case(*args)
    except Exception:
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def show_model(ax, mesh, values, title):
    """Plain matplotlib rendering -- avoids pyGIMLi's viewer, which needs
    an interactive backend and scipy.sparse."""
    cc = np.asarray(mesh.cellCenters(), dtype=float)
    v = np.log10(np.clip(np.asarray(values, dtype=float), 1e-3, None))
    ax.tricontourf(cc[:, 0], cc[:, 1], v, levels=24, cmap="Spectral_r")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title(title, fontsize=9)


def plot_envelope(df, out, rho=5.0, noise=0.01):
    sub = df[(np.isclose(df.rho_cav, rho)) & (np.isclose(df.noise, noise))]
    if sub.empty:
        print("  [skip] no data for that contrast/noise")
        return
    piv = sub.groupby(["top", "size"])["detected"].mean().unstack("size")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    X, Y = np.meshgrid(np.array(piv.columns), np.array(piv.index))
    im = ax.pcolormesh(X, Y, piv.values, cmap="viridis", vmin=0, vmax=1,
                       shading="nearest")
    try:                                    # guide lines: 50% and 90%
        cs = ax.contour(X, Y, piv.values, levels=[0.5, 0.9],
                        colors="white", linewidths=1.2)
        ax.clabel(cs, fmt="P=%.1f", fontsize=7)
    except Exception:
        pass
    ax.set_xlabel("cavity radius (m)")
    ax.set_ylabel("depth of cavity top (m)")
    ax.invert_yaxis()
    ax.set_title(f"Detection rate  (contrast {rho:g}x, noise {noise:.0%})")
    fig.colorbar(im, ax=ax, label="fraction detected")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_envelope.png"), dpi=200)
    plt.close(fig)


def plot_contrast(df, out, size=4.0, top=10.0):
    sub = df[(np.isclose(df["size"], size)) & (np.isclose(df.top, top))]
    if sub.empty:
        print("  [skip] no data for that size/depth")
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for noise, g in sub.groupby("noise"):
        st = g.groupby("rho_cav")["detected"].agg(["mean", "count"])
        p = st["mean"].values.astype(float)
        n = st["count"].values.astype(float)
        err = 1.96 * np.sqrt(np.clip(p * (1 - p), 0, None) / np.maximum(n, 1))
        ax.errorbar(st.index.values, p, yerr=err, fmt="o-", capsize=3,
                    label=f"noise {noise:.0%} (n={int(n[0])})")
    ax.set_xscale("log")
    ax.set_xlabel("cavity resistivity (ohm.m)")
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"radius {size:g} m, top {top:g} m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_contrast.png"), dpi=200)
    plt.close(fig)


def plot_null(null_df, out):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    sub = null_df[np.isclose(null_df["size"], 4.0)]
    for top, g in sub.groupby("top"):
        ax.hist(g.amp.dropna(), bins=25, alpha=0.5, label=f"top {top:g} m")
    ax.set_xlabel("recovered contrast (log10 units), noise only")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Null distribution -- no cavity present")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_null.png"), dpi=200)
    plt.close(fig)


def plot_examples(cfg, df, out):
    # a clear success: strongest recovered anomaly
    good = df[df.detected].sort_values("amp", ascending=False)
    # a clear failure: a big shallow target that SHOULD have been seen
    bad = df[(~df.detected) & (df["size"] >= 4.0) & (df.top <= 10.0)]
    if bad.empty:
        bad = df[~df.detected]
    if good.empty or bad.empty:
        print("  [skip] need both a success and a failure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for row, case in enumerate([good.iloc[0], bad.iloc[0]]):
        size, top = case["size"], case.top
        xc = cfg.x_start + (cfg.n_elec - 1) * cfg.spacing / 2.0
        zc = -(top + size)
        mesh = build_mesh(cfg, xc=xc, zc=zc, r=size)
        res = rho_map(cfg, mesh, case.rho_cav)
        show_model(axes[row, 0], mesh, res,
                   f"TRUE  r={size:g} m, top={top:g} m, "
                   f"{case.rho_cav:g} ohm.m")
        model, cells, mesh_inv = simulate_and_invert(cfg, mesh, case.rho_cav,
                                                     case.noise, int(case.seed),
                                                     make_scheme(cfg))
        show_model(axes[row, 1], mesh_inv, model,
                   f"INVERTED  amp={case.amp:.2f}, "
                   f"thr={case.thr:.2f}, detected={bool(case.detected)}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_examples.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# smoke test
# ----------------------------------------------------------------------
def smoke(cfg: Cfg, out: str):
    os.makedirs(out, exist_ok=True)
    # deliberately EASY case: big, shallow, low noise -> if this is not
    # recovered, the problem is a bug, not physics
    size, top, rho_cav, noise = 4.0, 5.0, 5.0, 0.005
    xc = cfg.x_start + (cfg.n_elec - 1) * cfg.spacing / 2.0
    zc = -(top + size)

    print(f"smoke: r={size} m, top={top} m, rho_cav={rho_cav}, "
          f"noise={noise:.0%}")
    t0 = time.time()
    mesh = build_mesh(cfg, xc=xc, zc=zc, r=size)
    print(f"  forward mesh: {mesh.cellCount()} cells")

    scheme = make_scheme(cfg)
    model, cells, mesh_inv = simulate_and_invert(cfg, mesh, rho_cav, noise,
                                                 7, scheme)
    amp, loc = amp_metric(model, cells, xc, zc, size)
    j = int(np.argmin(model))
    print(f"  inversion mesh: {len(model)} cells")
    print(f"  amp = {amp:.3f}, loc_err = {loc:.2f} m, "
          f"elapsed = {time.time() - t0:.1f}s")
    print(f"  global minimum at x={cells[j, 0]:.1f} m, z={cells[j, 1]:.1f} m "
          f"(truth: x={xc:.1f}, z={zc:.1f})")

    res = rho_map(cfg, mesh, rho_cav)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    show_model(axes[0], mesh, res, "TRUE model")
    show_model(axes[1], mesh_inv, model, "INVERTED")
    for ax in axes:                      # same view for easy comparison
        ax.set_xlim(-5, 53)
        ax.set_ylim(-35, 1)
    fig.tight_layout()
    path = os.path.join(out, "smoke.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved -> {path}")
    print("\nIf you can see a low-resistivity blob near the cavity, "
          "the pipeline works.  Now run --null.")


def run_serial(todo, csv_path, existing=None):
    """Plain loop -- no pickling, no workers, always works."""
    rows = []
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        try:
            r = run_case(*c)
        except Exception:
            traceback.print_exc()
            continue
        rows.append(r)
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"  {i}/{len(todo)}  elapsed {el / 60:.1f} min  "
                  f"eta {eta / 60:.1f} min", flush=True)
        if i % 25 == 0:                       # checkpoint
            df = pd.DataFrame(rows)
            if existing is not None and len(existing):
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(csv_path, index=False)
    return rows


def finalize(csv_path, thr, out, min_amp=0.0):
    """Attach thresholds, decide detection, draw the summary figures."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["size", "top", "rho_cav", "noise", "seed"])
    df = df.drop(columns=[c for c in ("thr", "detected") if c in df.columns])
    df = df.merge(thr, on=["noise", "size", "top"], how="left")
    n_missing = int(df.thr.isna().sum())
    if n_missing:
        print(f"  warning: {n_missing} cases have no threshold "
              f"(grid changed after calibration?)")
    df["detected"] = (df.amp > df.thr) & (df.amp > min_amp)
    df.to_csv(csv_path, index=False)
    plot_envelope(df, out)
    plot_contrast(df, out)
    print("  figures -> fig1_envelope.png, fig2_contrast.png")
    return df


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--null", type=int, default=50)
    ap.add_argument("--null-only", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--null-jobs", type=int, default=1)
    ap.add_argument("--max-cell", type=float, default=None,
                    help="max cell size of the inversion mesh (default 2.0)")
    ap.add_argument("--min-amp", type=float, default=0.0,
                    help="minimum effect size (log10) required for detection")
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard to run (0-based), for manual parallelism")
    ap.add_argument("--nshards", type=int, default=1,
                    help="total number of shards; run one terminal per shard")
    ap.add_argument("--merge", action="store_true",
                    help="combine results_shard*.csv and draw figures")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    quiet()
    os.makedirs(args.out, exist_ok=True)
    cfg = Cfg()
    if args.max_cell:
        cfg.inv_max_cell = args.max_cell
    scheme = make_scheme(cfg)

    if args.smoke:
        smoke(cfg, args.out)
        return

    # ---------------- figures only ----------------
    if args.plot or args.examples:
        csv_path = os.path.join(args.out, "results.csv")
        if not os.path.exists(csv_path):
            print(f"no {csv_path} yet -- run the sweep first")
            return
        df = pd.read_csv(csv_path)
        if args.plot:
            plot_envelope(df, args.out)
            plot_contrast(df, args.out)
            print("  figures -> fig1_envelope.png, fig2_contrast.png")
        if args.examples:
            plot_examples(cfg, df, args.out)
            print("  figures -> fig4_examples.png")
        return

    # ---------------- null calibration ----------------
    thr_path = os.path.join(args.out, "thresholds.csv")
    if args.null_only:
        print(f"null calibration: n={args.null} per noise level")
        null_df = make_null_table(cfg, args.null, scheme, SIZES, TOPS,
                                  NOISES, jobs=args.null_jobs)
        null_df.to_csv(os.path.join(args.out, "null.csv"), index=False)
        thr = thresholds_from_null(null_df)
        thr.to_csv(thr_path, index=False)
        plot_null(null_df, args.out)
        print(f"  saved -> {thr_path}")
        return

    if not os.path.exists(thr_path):
        print("thresholds missing.")
        print("run first:  python detectability.py --null 50 --null-only")
        return

    thr = pd.read_csv(thr_path)

    # ---------------- merge shards ----------------
    if args.merge:
        files = sorted(glob.glob(os.path.join(args.out, "results_shard*.csv")))
        if not files:
            print("no results_shard*.csv found")
            return
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out_csv = os.path.join(args.out, "results.csv")
        df.to_csv(out_csv, index=False)
        print(f"merged {len(files)} shards -> {len(df)} rows")
        finalize(out_csv, thr, args.out, args.min_amp)
        return

    # ---------------- sweep grid ----------------
    if args.fast:
        sizes, tops = [2.0, 4.0], [5.0, 15.0, 30.0]
        rhos, noises, seeds = [5.0], [0.01], [1]
    else:
        sizes, tops, rhos = SIZES, TOPS, RHOS
        noises, seeds = NOISES, SEEDS

    cfg_dict = asdict(cfg)
    cases = [(cfg_dict, s, t, r, n, sd)
             for s in sizes for t in tops
             for r in rhos for n in noises for sd in seeds]

    if args.nshards > 1:
        cases = [c for i, c in enumerate(cases)
                 if i % args.nshards == args.shard]
        csv_path = os.path.join(args.out, f"results_shard{args.shard}.csv")
    else:
        csv_path = os.path.join(args.out, "results.csv")
    done = set()
    existing = None
    if args.resume and os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done = {(round(r["size"], 3), round(r.top, 3),
                 round(r.rho_cav, 3), round(r.noise, 4), int(r.seed))
                for _, r in existing.iterrows()}
        print(f"resume: {len(done)} cases already done")

    todo = [c for c in cases
            if (round(c[1], 3), round(c[2], 3), round(c[3], 3),
                round(c[4], 4), c[5]) not in done]
    print(f"running {len(todo)} cases "
          f"({'parallel x' + str(args.jobs) if args.jobs > 1 else 'serial'})")

    t0 = time.time()
    rows = None
    if args.jobs > 1 and HAVE_JOBLIB:
        try:
            rows = Parallel(n_jobs=args.jobs, verbose=10)(
                delayed(safe_run_case)(*c) for c in todo)
            rows = [r for r in rows if r is not None]
        except Exception as e:
            print(f"  parallel failed ({type(e).__name__}: {e})")
            print("  falling back to serial -- or use --nshards/--shard")
            rows = None
    if rows is None:
        rows = run_serial(todo, csv_path, existing)

    if not rows:
        print("nothing produced")
        return

    df = pd.DataFrame(rows)
    if args.resume and os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        old = old.drop(columns=[c for c in ("thr", "detected")
                                if c in old.columns])
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["size", "top", "rho_cav",
                                        "noise", "seed"])

    df.to_csv(csv_path, index=False)
    print(f"  saved -> {csv_path}  ({len(df)} rows, "
          f"{(time.time() - t0) / 60:.1f} min)")

    if args.nshards > 1:
        print(f"  shard {args.shard}/{args.nshards} finished; "
              f"run the rest, then:  --merge")
        return

    finalize(csv_path, thr, args.out, args.min_amp)

    if args.examples:
        plot_examples(cfg, pd.read_csv(csv_path), args.out)
        print("  figures -> fig4_examples.png")


if __name__ == "__main__":
    main()
