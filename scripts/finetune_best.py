"""Pick the best run from the frozen SAM head sweep and launch full
end-to-end fine-tuning on it.

After the head sweep finishes (`pipeline=sam` over `net=sam,sam_narrow,
sam_wide,sam_deep`), this:

  1. Scans `runs/` for frozen-`sam` runs trained on the SAM-feature cache.
  2. Ranks them by `val_dice` — parsed from each run's `best-*.ckpt`
     filename, which is the highest-val_dice epoch `ModelCheckpoint` kept.
  3. Launches `scripts/train.py pipeline=sam_finetune ...` with the
     winner's head architecture and `init_from_checkpoint` pointing at its
     best checkpoint.

Usage:
    uv run python scripts/finetune_best.py
    uv run python scripts/finetune_best.py --latest 4       # scope to the last 4 runs
    uv run python scripts/finetune_best.py --run <run-name> # force a winner
    uv run python scripts/finetune_best.py --dry-run        # print the command, don't run it

Chain it straight after the sweep:
    uv run python scripts/train.py -m pipeline=sam ... && \\
        uv run python scripts/finetune_best.py
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import paths
from pato.experiments import list_runs, load_run

# --- the frozen head sweep this script consumes -----------------------------
SWEEP_PIPELINE = "sam"
SWEEP_DATASET = "nmsc-2x-sam-vit-base-1024"

# --- Phase 2 (full fine-tune) settings — edit here for different defaults ----
FINETUNE_DATASET = "nmsc-2x-sam-full-1024"
FINETUNE_LR = "cosine"
FINETUNE_OVERRIDES = [
    "lr.learning_rate=1.0e-4",
    "lr.scheduler.eta_min=1.0e-6",
    "lr.scheduler.T_max=10",
    "max_epochs=40",
]

# (widths tuple, blocks_per_stage) -> conf/net/ choice name. The four nets the
# sweep uses; anything else falls back to explicit head overrides.
KNOWN_NETS = {
    ((128, 64, 32, 16), 0): "sam",
    ((64, 32, 16, 8), 0): "sam_narrow",
    ((256, 128, 64, 32), 0): "sam_wide",
    ((128, 64, 32, 16), 2): "sam_deep",
}

_VAL_DICE_RE = re.compile(r"val_dice=(\d+\.\d+)")


def _pipeline_name(cfg: dict) -> str | None:
    p = cfg.get("pipeline")
    return p.get("name") if isinstance(p, dict) else p


def _is_frozen_sam_sweep_run(run) -> bool:
    cfg = run.config
    if _pipeline_name(cfg) != SWEEP_PIPELINE:
        return False
    pipeline = cfg.get("pipeline", {})
    if isinstance(pipeline, dict) and not pipeline.get("sam_frozen", False):
        return False
    # `dataset.train` is the post-train/val-split field; `dataset.dataset_root`
    # is the legacy field that older run configs wrote. Check both so this
    # script keeps picking up sweep runs across the migration.
    ds = cfg.get("dataset", {})
    train_cache = str(ds.get("train", ds.get("dataset_root", "")))
    return Path(train_cache).name == SWEEP_DATASET


def _val_dice(run) -> float | None:
    ckpt = run.best_checkpoint()
    if ckpt is None:
        return None
    m = _VAL_DICE_RE.search(ckpt.name)
    return float(m.group(1)) if m else None


def find_sweep_runs(latest: int) -> list:
    # list_runs() returns oldest-first / newest-last (by mtime), so the
    # tail is genuinely the most recent N runs.
    runs = [load_run(paths.runs / n) for n in list_runs(paths.runs)]
    runs = [r for r in runs if _is_frozen_sam_sweep_run(r)]
    return runs[-latest:] if latest else runs


def _net_overrides(run) -> list[str]:
    head = run.config["net"]["head"]
    widths = tuple(head["widths"])
    blocks = head.get("blocks_per_stage", 0)
    name = KNOWN_NETS.get((widths, blocks))
    if name:
        return [f"net={name}"]
    # Unknown widths/blocks combo — rebuild it via explicit head overrides.
    widths_str = ",".join(str(w) for w in widths)
    return [
        "net=sam",
        f"net.head.widths=[{widths_str}]",
        f"net.head.blocks_per_stage={blocks}",
    ]


def build_finetune_command(run) -> list[str]:
    ckpt = run.best_checkpoint()
    return [
        "uv", "run", "python", "scripts/train.py",
        "pipeline=sam_finetune",
        f"dataset={FINETUNE_DATASET}",
        *_net_overrides(run),
        f"lr={FINETUNE_LR}",
        *FINETUNE_OVERRIDES,
        # Single-quote the path: Lightning ckpt filenames contain '=' which
        # Hydra's override parser would otherwise mis-split.
        f"init_from_checkpoint='{ckpt.resolve()}'",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--latest", type=int, default=4,
        help="consider only the N most recent matching runs (0 = all). default: 4",
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="force this run name as the winner (skip auto-pick)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the fine-tune command but don't execute it",
    )
    args = parser.parse_args()

    if args.run:
        winner = load_run(paths.runs / args.run)
        print(f"Forced winner: {winner.name}")
    else:
        runs = find_sweep_runs(args.latest)
        if not runs:
            print(
                f"No frozen-sam runs on {SWEEP_DATASET} found under {paths.runs}",
                file=sys.stderr,
            )
            return 1
        scored = [(r, _val_dice(r)) for r in runs]
        for r, s in scored:
            if s is None:
                print(f"  (skipped {r.name}: no best-*.ckpt)", file=sys.stderr)
        # val_dice: higher is better → sort descending.
        ranked = sorted(
            ((r, s) for r, s in scored if s is not None),
            key=lambda rs: rs[1],
            reverse=True,
        )
        if not ranked:
            print("No runs had a best checkpoint to score.", file=sys.stderr)
            return 1
        print(f"Ranked {len(ranked)} run(s) by val_dice (higher = better):")
        for i, (r, s) in enumerate(ranked):
            print(f"  {s:.4f}  {r.name}{'   <- winner' if i == 0 else ''}")
        winner, _ = ranked[0]

    cmd = build_finetune_command(winner)
    print()
    print("Fine-tune command:")
    print("  " + " ".join(cmd))
    if args.dry_run:
        print("\n(--dry-run: not executing)")
        return 0
    print()
    return subprocess.run(cmd, cwd=_PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
