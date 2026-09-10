"""Tests for the SWHD_1D (1D shallow water) solver."""
import numpy as np

import spooky as sp
from spooky.solvers import SWHD_1D

# Shared parameters — small grid so tests run fast
LX = 2 * np.pi
NX = 64
DT = 1e-3
G  = 1.0
H0 = 1.0   # mean depth


def _make_solver(hb=None, rkord=2):
    """hb may be None or a callable grid -> array."""
    grid   = sp.Grid1D(LX, NX, DT)
    if hb is not None:
        hb = hb(grid)
    solver = SWHD_1D(grid, g=G, hb=hb, rkord=rkord)
    return grid, solver


def _ic(grid, amp=0.1):
    """Zero velocity and a single-mode free-surface perturbation."""
    uu = np.zeros_like(grid.xx)
    hh = H0 + amp * np.cos(2 * grid.xx)
    return [uu, hh]


def _bump(grid):
    """Smooth bottom topography, mean zero."""
    return 0.2 * np.cos(3 * grid.xx)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_default_bottom_is_flat():
    grid, solver = _make_solver()
    assert solver.hb.shape == grid.shape
    assert np.all(solver.hb == 0.0)


def test_mass_is_conserved_over_topography():
    """<h - hb> is an exact invariant of the discrete scheme.

    The h equation is a pure x-derivative, so its k=0 mode never changes.
    """
    grid, solver = _make_solver(hb=_bump)
    fields = _ic(grid)
    ffields0 = [grid.forward(ff) for ff in fields]
    m0 = solver.mass(ffields0)

    fields = solver.evolve(fields, T=1.0, write_outputs=False)
    m1 = solver.mass([grid.forward(ff) for ff in fields])

    assert abs(m1 - m0) < 1e-13


def test_energy_is_conserved_over_topography():
    """<(h-hb) u^2/2 + g h^2/2> is conserved up to time-stepping error."""
    grid, solver = _make_solver(hb=_bump)
    fields = _ic(grid)
    e0 = solver.energy([grid.forward(ff) for ff in fields])

    fields = solver.evolve(fields, T=1.0, write_outputs=False)
    e1 = solver.energy([grid.forward(ff) for ff in fields])

    assert abs(e1 - e0) / e0 < 1e-6


def test_linear_gravity_wave_speed():
    """A small perturbation over a flat bottom is a standing wave with
    frequency omega = sqrt(g H0) k, so h(x,t) = H0 + a cos(kx) cos(omega t).
    """
    amp = 1e-3
    k   = 2
    T   = 1.0
    grid, solver = _make_solver()
    fields = solver.evolve(_ic(grid, amp=amp), T=T, write_outputs=False)

    omega  = np.sqrt(G * H0) * k
    h_lin  = H0 + amp * np.cos(k * grid.xx) * np.cos(omega * T)
    u_lin  = amp * np.sqrt(G / H0) * np.sin(k * grid.xx) * np.sin(omega * T)

    # Nonlinear corrections are O(amp^2)
    np.testing.assert_allclose(fields[1], h_lin, atol=5 * amp**2)
    np.testing.assert_allclose(fields[0], u_lin, atol=5 * amp**2)


def test_outputs_and_balance(tmp_path):
    """outs writes uu/hh files to opath, balance writes 3 columns to bpath."""
    grid, solver = _make_solver()
    opath = tmp_path / "out"
    bpath = tmp_path / "bal"
    opath.mkdir(); bpath.mkdir()

    nsteps = 10
    solver.evolve(_ic(grid), T=nsteps * DT, bstep=5, ostep=5,
                  bpath=str(bpath), opath=str(opath))

    for step in (0, 5, 10):
        assert (opath / f"uu.{step:04}.npy").exists()
        assert (opath / f"hh.{step:04}.npy").exists()

    uu, hh = solver.load_fields(str(opath), nsteps)
    assert uu.shape == hh.shape == grid.shape

    bal = np.loadtxt(bpath / "balance.dat")
    assert bal.shape[1] == 3
    np.testing.assert_allclose(bal[0, 0], 0.0)
    np.testing.assert_allclose(bal[-1, 0], nsteps * DT)
