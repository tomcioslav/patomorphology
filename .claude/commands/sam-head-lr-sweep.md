---
description: Train all sam_head net variants across an LR sweep (Hydra multirun)
argument-hint: "[net_variants] [lr_choices]   e.g. 'sam_head,sam_head_wide' 'constant_1e3,cosine'"
allowed-tools: Bash
---

Run a Hydra multirun sweep over `sam_head` architecture variants × LR schedules.

Arguments (positional, both optional):
  1. Comma-separated `net` choices.
     Default: `sam_head,sam_head_narrow,sam_head_wide,sam_head_deep`
  2. Comma-separated `lr` choices.
     Default: `constant_1e3,constant_3e4,constant_1e4`

Parse `$ARGUMENTS` into the two positional values (defaults above when not
supplied), then run from the repo root:

```
uv run python scripts/train.py -m \
    pipeline=sam_head \
    dataset=nmsc-2x-sam-vit-base-1024 \
    net=<net_variants> \
    lr=<lr_choices>
```

Stream the output. After the sweep, list new run dirs under `runs/` so the
user can open them in TensorBoard.
