import os, re, hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (13, 5)
plt.rcParams['axes.grid'] = True

# ── Color helpers ────────────────────────────────────────────────────────────
NAMED_COLORS = {
    'red': '#d62728', 'green': '#2ca02c', 'blue': '#1f77b4', 'yellow': '#bcbd22',
    'orange': '#ff7f0e', 'purple': '#9467bd', 'brown': '#8c564b', 'pink': '#e377c2',
    'gray': '#7f7f7f', 'grey': '#7f7f7f', 'black': '#111111', 'white': '#f5f5f5',
    'cyan': '#17becf', 'magenta': '#d33682',
}
FALLBACK_PALETTE = [
    '#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2',
    '#ff9da6', '#9d755d', '#bab0ab', '#6f4c9b', '#2f4b7c', '#a05195',
]
_COLOR_KEYS = sorted(NAMED_COLORS, key=len, reverse=True)

def _lighten(hex_color: str, amount=0.35) -> str:
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex(rgb + (1 - rgb) * amount)

def color_for_filename(name: str) -> Tuple[str, str]:
    """Return (base_hex, display_hex) derived from the filename stem."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    variant = re.search(r'_(\d+)$', stem)
    for key in _COLOR_KEYS:
        if re.search(rf'(^|[^a-z]){re.escape(key)}([^a-z]|$)', stem):
            base = NAMED_COLORS[key]
            disp = _lighten(base, 0.25 + 0.08 * min(int(variant.group(1)), 5)) if variant else base
            return base, disp
    idx = int(hashlib.md5(stem.encode()).hexdigest(), 16) % len(FALLBACK_PALETTE)
    base = FALLBACK_PALETTE[idx]
    return base, base
# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class PolygonRecord:
    name: str
    pts3d: np.ndarray
    base_color: str
    display_color: str
    # populated by analyze()
    valid: bool = False          # planar + non-degenerate
    planar: bool = False
    degenerate: bool = False
    self_intersecting: bool = False
    convex: Optional[bool] = None
    area: Optional[float] = None
    max_planarity_dev: Optional[float] = None
    pts2d: Optional[np.ndarray] = None
    plane_origin: Optional[np.ndarray] = None
    basis_u: Optional[np.ndarray] = None
    basis_v: Optional[np.ndarray] = None
    messages: List[str] = field(default_factory=list)

# ── Geometry helpers ─────────────────────────────────────────────────────────

def parse_polygon_text(text: str, name: str) -> np.ndarray:
    """Parse whitespace/comma/semicolon-separated 3D points from text."""
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        s = re.sub(r'[\(\)\[\]]', ' ', line.strip()).strip()
        if not s or s.startswith('#'):
            continue
        parts = re.split(r'[\s,;]+', s)
        if len(parts) != 3:
            raise ValueError(f'{name}: line {i} does not have exactly 3 coords')
        rows.append(tuple(map(float, parts)))
    if len(rows) < 3:
        raise ValueError(f'{name}: need at least 3 points, got {len(rows)}')
    pts = np.array(rows, dtype=float)
    # Remove duplicate closing point if present
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    return pts


def fit_plane(pts3d: np.ndarray, eps=1e-6) -> dict:
    """Project 3D points onto their best-fit plane via SVD."""
    origin = pts3d.mean(axis=0)
    centered = pts3d - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u, v, normal = vh[0], vh[1], vh[2]
    max_dev = float(np.max(np.abs(centered @ normal)))
    pts2d = np.column_stack([centered @ u, centered @ v])
    return dict(origin=origin, u=u, v=v, normal=normal,
                pts2d=pts2d, max_dev=max_dev, planar=max_dev <= eps)


def signed_area(pts2d: np.ndarray) -> float:
    x, y = pts2d[:, 0], pts2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def has_self_intersections(pts2d: np.ndarray, eps=1e-9) -> bool:
    n = len(pts2d)
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def on_seg(a, b, c):
        return (min(a[0],b[0])-eps <= c[0] <= max(a[0],b[0])+eps and
                min(a[1],b[1])-eps <= c[1] <= max(a[1],b[1])+eps)
    for i in range(n):
        a1, a2 = pts2d[i], pts2d[(i+1)%n]
        for j in range(i+2, n):
            if i == 0 and j == n-1:
                continue
            b1, b2 = pts2d[j], pts2d[(j+1)%n]
            o1,o2 = orient(a1,a2,b1), orient(a1,a2,b2)
            o3,o4 = orient(b1,b2,a1), orient(b1,b2,a2)
            if (o1*o2 < -eps and o3*o4 < -eps):
                return True
            for o,a,b,c in [(o1,a1,a2,b1),(o2,a1,a2,b2),(o3,b1,b2,a1),(o4,b1,b2,a2)]:
                if abs(o) <= eps and on_seg(a,b,c):
                    return True
    return False


def is_convex(pts2d: np.ndarray, eps=1e-9) -> bool:
    n = len(pts2d)
    v = np.roll(pts2d, -1, axis=0) - pts2d          # edge vectors
    z = v[:,0] * np.roll(v[:,1], -1) - v[:,1] * np.roll(v[:,0], -1)  # cross product z-components
    signs = z[np.abs(z) > eps]
    return bool(len(signs) > 0) and (bool(np.all(signs > 0)) or bool(np.all(signs < 0)))
# ── Polygon analysis ─────────────────────────────────────────────────────────

def analyze(pts3d: np.ndarray, name: str, eps_plane=1e-6) -> PolygonRecord:
    base, disp = color_for_filename(name)
    rec = PolygonRecord(name=name, pts3d=pts3d, base_color=base, display_color=disp)

    fp = fit_plane(pts3d, eps=eps_plane)
    rec.max_planarity_dev = fp['max_dev']
    rec.planar = fp['planar']
    rec.plane_origin, rec.basis_u, rec.basis_v = fp['origin'], fp['u'], fp['v']
    rec.pts2d = fp['pts2d']

    if not rec.planar:
        rec.messages.append(f'Non-planar: max deviation = {rec.max_planarity_dev:.3e}')
        return rec

    area = signed_area(rec.pts2d)
    rec.area = abs(area)
    if rec.area < 1e-9:
        rec.degenerate = True
        rec.messages.append('Degenerate: near-zero area')
        return rec

    rec.self_intersecting = has_self_intersections(rec.pts2d)
    rec.convex = is_convex(rec.pts2d) if not rec.self_intersecting else False
    rec.valid = True
    rec.messages.append('Simple polygon.' if not rec.self_intersecting else 'Self-intersecting polygon.')
    rec.messages.append('Convex.' if rec.convex else 'Non-convex.')
    return rec


# ── Congruence test ───────────────────────────────────────────────────────────

def _procrustes_error(A: np.ndarray, B: np.ndarray) -> float:
    """Max per-vertex error after optimal rotation+translation of B onto A."""
    ca, cb = A.mean(0), B.mean(0)
    H = (B - cb).T @ (A - ca)
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    return float(np.max(np.linalg.norm((B - cb) @ R + ca - A, axis=1)))


def test_congruence(rec_a: PolygonRecord, rec_b: PolygonRecord,
                    atol=0.5, allow_self_intersecting=True) -> dict:
    """
    Test if two polygons are congruent by trying all cyclic shifts and reversals,
    then fitting with an orthogonal (rotation + translation) transform.
    Returns a result dict with keys: congruent, reason, and (if congruent)
    shift, reversed_order, used_reflection, max_err, B2_fit.
    """
    if not rec_a.valid or not rec_b.valid:
        return dict(congruent=False, reason='One or both polygons are invalid/non-planar.')
    if not allow_self_intersecting and (rec_a.self_intersecting or rec_b.self_intersecting):
        return dict(congruent=False, reason='Self-intersecting polygons excluded.')
    if len(rec_a.pts2d) != len(rec_b.pts2d):
        return dict(congruent=False, reason=f'Different vertex count ({len(rec_a.pts2d)} vs {len(rec_b.pts2d)}).')

    A, B = rec_a.pts2d, rec_b.pts2d
    n = len(A)

    # Quick pre-filter: sorted side-lengths must match
    la = np.sort(np.linalg.norm(np.roll(A, -1, 0) - A, axis=1))
    lb = np.sort(np.linalg.norm(np.roll(B, -1, 0) - B, axis=1))
    if not np.allclose(la, lb, atol=atol * 2, rtol=0):
        return dict(congruent=False, reason='Side lengths do not match.')

    best_err, best_shift, best_rev = float('inf'), 0, False
    for reverse in (False, True):
        Bc = B[::-1] if reverse else B
        for shift in range(n):
            Bs = np.roll(Bc, -shift, axis=0)
            err = _procrustes_error(A, Bs)
            if err < best_err:
                best_err, best_shift, best_rev = err, shift, reverse

    if best_err > atol:
        return dict(congruent=False, reason=f'Best fit error {best_err:.3f} exceeds tolerance {atol}.')

    # Compute final transform for the best alignment
    Bc = B[::-1] if best_rev else B
    Bs = np.roll(Bc, -best_shift, axis=0)
    ca, cb = A.mean(0), Bs.mean(0)
    H = (Bs - cb).T @ (A - ca)
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    B_fit = (Bs - cb) @ R + ca

    return dict(
        congruent=True,
        reason=f'Max fit error: {best_err:.3e}',
        shift=best_shift,
        reversed_order=best_rev,
        used_reflection=bool(np.linalg.det(R) < 0),
        max_err=best_err,
        B2_fit=B_fit,
        B2_reordered=Bs,
    )

# ── Plotting ─────────────────────────────────────────────────────────────────

def _close(pts):
    return np.vstack([pts, pts[0]])

def plot_pair(rec_a: PolygonRecord, rec_b: PolygonRecord, result: dict = None):
    fig = plt.figure(figsize=(13, 5))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    # 2D projected view
    a2, b2 = _close(rec_a.pts2d), _close(rec_b.pts2d)
    ax2d.plot(a2[:,0], a2[:,1], '-o', color=rec_a.display_color, label=rec_a.name)
    ax2d.plot(b2[:,0], b2[:,1], '-o', color=rec_b.display_color, alpha=0.65, label=rec_b.name)
    if result and result['congruent']:
        bf = _close(result['B2_fit'])
        ax2d.plot(bf[:,0], bf[:,1], '--s', color='black', alpha=0.7, label=f'{rec_b.name} (transformed)')
    ax2d.set_title('Projected 2D'); ax2d.axis('equal'); ax2d.legend(fontsize=8)

    # 3D view
    a3, b3 = _close(rec_a.pts3d), _close(rec_b.pts3d)
    ax3d.plot(a3[:,0], a3[:,1], a3[:,2], '-o', color=rec_a.display_color, label=rec_a.name)
    ax3d.plot(b3[:,0], b3[:,1], b3[:,2], '-o', color=rec_b.display_color, alpha=0.65, label=rec_b.name)
    if result and result['congruent']:
        # Lift fitted 2D points back into rec_a's 3D plane
        b3fit = rec_a.plane_origin + np.outer(result['B2_fit'][:,0], rec_a.basis_u) \
                                   + np.outer(result['B2_fit'][:,1], rec_a.basis_v)
        b3fit = _close(b3fit)
        ax3d.plot(b3fit[:,0], b3fit[:,1], b3fit[:,2], '--s', color='black', alpha=0.7,
                  label=f'{rec_b.name} (transformed)')
    ax3d.set_title('3D view'); ax3d.legend(fontsize=8)
    plt.tight_layout(); 
    return fig