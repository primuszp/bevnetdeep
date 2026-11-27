"""Airborne scan-line degradation for synthetic LiDAR scenes.

Turns the dense synthetic point cloud into sparse, striped evidence that
mimics real aerial surveys (YUTO / ECLAIR / ISPRS), so a model trained on
(degraded input -> perfect GT labels) learns to infer full footprints and
OBBs from exactly the evidence pattern the real weak domains provide.

Profile priors come from research/bev/measure_scan_pattern.py on bev_dataset_v4
(2026-07-03, cell 0.2 m):

  source   stripe p50   occupied   veh px/box p50   fill in box
  yuto        0.50 m       0.17          21             0.32
  eclair      0.50 m       0.57          36             0.43
  isprs       1.26 m       0.13          13             0.22

Model: each scene gets 1-2 scan passes; a pass keeps points within a beam
half-width of parallel scan lines (random heading + phase), then thins the
survivors with a keep probability. Passes are OR-ed like overlapping flight
strips. All classes (ground, vehicles, clutter) share the pattern, as in a
real survey. Domain randomisation is deliberately wider than the measured
p25-p75 so the trained model does not lock onto one scanner geometry.

Beyond straight uniform stripes, a pass can also have:
  * line wobble    — oscillating scanner mirror: lines drift sinusoidally,
  * segment dropout — whole along-line chunks missing (occlusion, strip edge).
The *_wide and *_patchy YUTO variants exercise the failure mode observed on
real tiles (locally >1 m stripe gaps splitting one car into crosswise
fragments) so the teacher and the label benchmark both see it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MIX_PROFILES",
    "PROFILES",
    "ScanProfile",
    "estimate_stripe_spacing_m",
    "sample_passes",
    "scan_keep_mask",
]


def estimate_stripe_spacing_m(occ: np.ndarray, cell: float,
                              win_px: int = 512) -> float | None:
    """Median scan-line spacing of an occupancy grid, or None if not striped.

    2-D FFT peak per moderately filled window (same method as
    research/bev/measure_scan_pattern.py). Lets the label pipeline adapt its
    stripe-bridging closing radius to each tile instead of a global constant:
    an under-bridged tile splits one vehicle into per-stripe fragments that
    then get fitted crosswise.
    """
    H, W = occ.shape
    if min(H, W) < 64:
        return None
    if min(H, W) < win_px:
        win_px = 1 << (min(H, W).bit_length() - 1)
    spacings = []
    for r0 in range(0, H - win_px + 1, win_px):
        for c0 in range(0, W - win_px + 1, win_px):
            win = occ[r0:r0 + win_px, c0:c0 + win_px].astype(np.float32)
            fill = float(win.mean())
            if fill < 0.02 or fill > 0.60:
                continue
            f = np.abs(np.fft.fftshift(np.fft.fft2(win - win.mean())))
            c = win_px // 2
            yy, xx = np.mgrid[0:win_px, 0:win_px]
            rad = np.hypot(yy - c, xx - c)
            band = (rad >= max(2, win_px * cell / 20.0)) & \
                   (rad <= min(win_px // 2 - 1, win_px * cell / 0.4))
            if not band.any():
                continue
            i = int(np.argmax(np.where(band, f, 0)))
            py, px = divmod(i, win_px)
            if f[py, px] < 8.0 * (np.median(f[band]) + 1e-9):
                continue
            spacings.append(win_px * cell / float(rad[py, px]))
    return float(np.median(spacings)) if spacings else None


@dataclass(frozen=True)
class ScanProfile:
    """Randomisation ranges for one target domain (all lengths in metres)."""
    name: str
    spacing: tuple[float, float]      # scan-line pitch
    beam_width: tuple[float, float]   # swath a line illuminates
    along_keep: tuple[float, float]   # survivor thinning probability
    second_pass_prob: float           # chance of an extra crossing strip
    second_pass_keep_scale: float = 0.6   # second strip is usually thinner
    wobble_amp: tuple[float, float] = (0.0, 0.0)     # line drift amplitude
    wobble_wavelength: tuple[float, float] = (20.0, 60.0)
    seg_drop_prob: tuple[float, float] = (0.0, 0.0)  # whole-chunk dropout
    seg_len: tuple[float, float] = (6.0, 20.0)       # dropout chunk length


PROFILES: dict[str, ScanProfile] = {
    "yuto":   ScanProfile("yuto",   spacing=(0.40, 0.80), beam_width=(0.10, 0.22),
                          along_keep=(0.35, 0.75), second_pass_prob=0.35,
                          wobble_amp=(0.0, 0.15), seg_drop_prob=(0.0, 0.15)),
    "eclair": ScanProfile("eclair", spacing=(0.35, 0.65), beam_width=(0.18, 0.32),
                          along_keep=(0.55, 0.95), second_pass_prob=0.65,
                          wobble_amp=(0.0, 0.10)),
    "isprs":  ScanProfile("isprs",  spacing=(0.90, 1.70), beam_width=(0.10, 0.20),
                          along_keep=(0.25, 0.60), second_pass_prob=0.15),
    # YUTO-like stress variants: the real failure cases are tiles whose LOCAL
    # stripe gap exceeds ~1 m and tiles with patchy, interrupted lines.
    "yuto_wide":   ScanProfile("yuto_wide", spacing=(0.70, 1.60),
                               beam_width=(0.10, 0.22),
                               along_keep=(0.30, 0.65), second_pass_prob=0.25,
                               wobble_amp=(0.05, 0.25),
                               seg_drop_prob=(0.05, 0.25)),
    "yuto_patchy": ScanProfile("yuto_patchy", spacing=(0.40, 1.00),
                               beam_width=(0.10, 0.20),
                               along_keep=(0.30, 0.70), second_pass_prob=0.30,
                               wobble_amp=(0.05, 0.30),
                               seg_drop_prob=(0.25, 0.55), seg_len=(3.0, 12.0)),
}

# Scene-level draw list for --scan-profile mix: the YUTO stress variants get
# their own slots so roughly 40% of scenes exercise sparse/patchy scanning.
MIX_PROFILES: tuple[str, ...] = ("yuto", "eclair", "isprs", "yuto_wide", "yuto_patchy")


@dataclass(frozen=True)
class ScanPass:
    angle_rad: float
    spacing_m: float
    phase_m: float
    beam_half_m: float
    keep_prob: float
    wobble_amp_m: float
    wobble_wl_m: float
    wobble_phase: float
    seg_drop_prob: float
    seg_len_m: float
    salt: int


def sample_passes(rng: np.random.Generator, profile: ScanProfile) -> list[ScanPass]:
    """Draw the concrete scan geometry of one scene from a profile."""
    passes = [_one_pass(rng, profile, keep_scale=1.0)]
    if rng.random() < profile.second_pass_prob:
        passes.append(_one_pass(rng, profile,
                                keep_scale=profile.second_pass_keep_scale))
    return passes


def _one_pass(rng, profile: ScanProfile, keep_scale: float) -> ScanPass:
    spacing = float(rng.uniform(*profile.spacing))
    return ScanPass(
        angle_rad=float(rng.uniform(0, np.pi)),
        spacing_m=spacing,
        phase_m=float(rng.uniform(0, spacing)),
        beam_half_m=0.5 * float(rng.uniform(*profile.beam_width)),
        keep_prob=float(np.clip(rng.uniform(*profile.along_keep) * keep_scale,
                                0.02, 1.0)),
        wobble_amp_m=float(rng.uniform(*profile.wobble_amp)),
        wobble_wl_m=float(rng.uniform(*profile.wobble_wavelength)),
        wobble_phase=float(rng.uniform(0, 2 * np.pi)),
        seg_drop_prob=float(rng.uniform(*profile.seg_drop_prob)),
        seg_len_m=float(rng.uniform(*profile.seg_len)),
        salt=int(rng.integers(1, 2**31 - 1)),
    )


def _segment_hash01(line_idx: np.ndarray, seg_idx: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic pseudo-random in [0,1) per (scan line, along-line chunk)."""
    h = (line_idx.astype(np.int64) * 73856093
         ^ seg_idx.astype(np.int64) * 19349663
         ^ np.int64(salt)) & 0x7FFFFFFF
    # xorshift-style scramble to decorrelate neighbouring ids
    h = (h ^ (h >> 13)) * 1274126177 & 0x7FFFFFFF
    return (h % 104729) / 104729.0


def scan_keep_mask(xy: np.ndarray, rng: np.random.Generator,
                   passes: list[ScanPass]) -> np.ndarray:
    """Boolean keep mask for (N, 2) point coordinates under the scan passes."""
    xy = np.asarray(xy, np.float64)
    keep = np.zeros(len(xy), bool)
    for p in passes:
        ca, sa = np.cos(p.angle_rad), np.sin(p.angle_rad)
        u = xy[:, 0] * ca + xy[:, 1] * sa          # across scan lines
        v = -xy[:, 0] * sa + xy[:, 1] * ca         # along scan lines
        if p.wobble_amp_m > 0:
            u = u + p.wobble_amp_m * np.sin(
                2 * np.pi * v / max(p.wobble_wl_m, 1e-6) + p.wobble_phase)
        shifted = u + p.phase_m
        line_idx = np.floor(shifted / p.spacing_m + 0.5)
        d = np.abs(shifted - line_idx * p.spacing_m)  # dist to nearest line
        on_line = d <= p.beam_half_m
        if p.seg_drop_prob > 0:
            seg_idx = np.floor(v / max(p.seg_len_m, 1e-6))
            on_line &= _segment_hash01(line_idx, seg_idx, p.salt) >= p.seg_drop_prob
        if p.keep_prob < 1.0:
            on_line &= rng.random(len(xy)) < p.keep_prob
        keep |= on_line
    return keep
