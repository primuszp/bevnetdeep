"""Console progress and epoch summary helpers for BEV training."""

from __future__ import annotations

import os
import sys


# Live, in-place progress (carriage-return bar) needs an interactive terminal.
# Honor FORCE_COLOR=1 so runs launched from a script/wrapper still get the live
# bar, and NO_COLOR (https://no-color.org) to opt out. When off, the trainer
# emits sparse plain progress lines instead of spamming the log.
def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


USE_COLOR = _use_color()


def color_text(code: str, text: str) -> str:
    """Wrap text in an ANSI color/style code when stdout is interactive."""
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def progress_bar(frac: float, width: int = 26) -> str:
    """Return a compact fixed-width progress bar."""
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    return "#" * n + "." * (width - n)


def format_hms(secs: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    secs = int(max(secs, 0))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def print_epoch_summary(
    ep,
    total_epochs,
    train_parts,
    vl,
    best,
    improved,
    weights,
    lr_now,
    bad,
    patience,
    secs,
    train_yaw_mae,
    val_yaw_mae,
    val_det=None,
    val_det_ema=None,
    nonfinite_grad_batches: int = 0,
    nonfinite_loss_batches: int = 0,
    nonfinite_grad_only_batches: int = 0,
    retried_nonfinite_loss_batches: int = 0,
) -> None:
    """Print the compact per-epoch metrics; full details stay in CSV/plots."""
    thm, treg, tseg, _tcons, _torient, _tmoment, _tzheight, _tshadow, _tgreg, ttot = train_parts
    vd = vl if val_det is None else val_det
    ve = vd if val_det_ema is None else val_det_ema

    def fmt(v: float) -> str:
        try:
            fv = float(v)
        except Exception:
            return "nan"
        return f"{fv:.3f}" if abs(fv) < 1e9 else str(fv)

    head = color_text(
        "1", f"epoch {ep + 1:>3}/{total_epochs}  {format_hms(secs):>7s}  lr {lr_now:.1e}"
    )
    val = f"val_det {fmt(vd)}  ema {fmt(ve)}  best {fmt(best)}  val_loss {fmt(vl)}"
    if improved:
        val = color_text("32;1", val + "  saved best")
    else:
        delta = ve - best if all(map(lambda x: x == x, (ve, best))) else float("nan")
        val = val + f"  patience {bad}/{patience}  delta {fmt(delta)}"
    diag = ""
    if nonfinite_grad_batches or retried_nonfinite_loss_batches:
        diag = (
            f"\n  numeric: retry={retried_nonfinite_loss_batches} "
            f"skip={nonfinite_grad_batches} "
            f"(loss={nonfinite_loss_batches}, grad={nonfinite_grad_only_batches})"
        )
    print(
        f"{head}\n"
        f"  train: total {fmt(ttot)}  hm {fmt(thm)}  reg {fmt(treg)}  seg {fmt(tseg)}\n"
        f"  valid: {val}{diag}",
        flush=True,
    )
