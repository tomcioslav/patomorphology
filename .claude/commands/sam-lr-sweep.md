---
description: Train all SAM head variants across an LR sweep (Hydra multirun)
argument-hint: "[net_variants] [lr_choices]   e.g. 'sam,sam_wide' 'constant_1e3,cosine'"
allowed-tools: Bash
---

Run a Hydra multirun sweep over `sam` head-architecture variants × LR
schedules. Uses the frozen-encoder regime (`pipeline=sam` defaults to
`sam_frozen=true`), training the head against the SAM-feature cache.

Arguments (positional, both optional):
  1. Comma-separated `net` choices. Default: `sam,sam_narrow,sam_wide,sam_deep`
  2. Comma-separated `lr` choices. Default: `constant_1e3,constant_3e4,constant_1e4`

Parse `$ARGUMENTS` into the two positional values (defaults above when not
supplied), then run from the repo root:

```
uv run python scripts/train.py -m \
    pipeline=sam \
    dataset=nmsc-2x-sam-vit-base-1024 \
    net=<net_variants> \
    lr=<lr_choices>
```

Stream the output. After the sweep, list new run dirs under `runs/` so the
user can compare them in the W&B project at wandb.ai/<entity>/patomorphology.
