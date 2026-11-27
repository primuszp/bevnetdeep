"""Version-independent command dispatch for BEV training and inference."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

Entrypoint = Callable[[Sequence[str] | None], None]


def _extra_args(args: Sequence[str] | None) -> list[str]:
    return list(sys.argv[1:] if args is None else args)


def train(args: Sequence[str] | None = None) -> None:
    """Train a BEV model using explicitly supplied training arguments."""
    from .training.train_bev import main as train_main

    train_main(_extra_args(args))


def evaluate(args: Sequence[str] | None = None) -> None:
    """Evaluate the weights supplied with ``--weights``."""
    from .evaluation import main as evaluate_main

    evaluate_main(_extra_args(args))


def jeli(args: Sequence[str] | None = None) -> None:
    """Run Jeli-pair inference using the weights supplied with ``--weights``."""
    try:
        from .jeli import main as jeli_main
    except ModuleNotFoundError as exc:
        if exc.name in {"laspy", "matplotlib", "torch"}:
            raise ModuleNotFoundError(
                f"Missing dependency {exc.name!r}. Install BevNetDeep with "
                "'python -m pip install .' from the repository root."
            ) from exc
        raise

    jeli_main(_extra_args(args))


def detect(args: Sequence[str] | None = None) -> None:
    """Classify one LAS/LAZ cloud and run the supplied BEV weights."""
    from .pointcloud import main as pointcloud_main

    pointcloud_main(_extra_args(args))


COMMANDS: dict[str, Entrypoint] = {
    "train": train,
    "evaluate": evaluate,
    "jeli": jeli,
    "detect": detect,
}


def run(command: str, args: Sequence[str] = ()) -> None:
    """Run a named workflow without mutating process-global ``sys.argv``."""
    try:
        entrypoint = COMMANDS[command]
    except KeyError as exc:
        raise ValueError(
            f"unknown BEV workflow {command!r}; choose from {', '.join(COMMANDS)}"
        ) from exc
    entrypoint(args)


def main(args: Sequence[str] | None = None) -> None:
    argv = _extra_args(args)
    if argv in (["-h"], ["--help"]):
        print(f"usage: bordereye-bev [{' | '.join(COMMANDS)}] [options]")
        return
    if not argv or argv[0] not in COMMANDS:
        names = " | ".join(COMMANDS)
        raise SystemExit(f"usage: python -m bordereye.ml.bev.workflow [{names}] [options]")
    run(argv[0], argv[1:])


if __name__ == "__main__":
    main()
