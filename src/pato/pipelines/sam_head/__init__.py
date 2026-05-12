"""SAM-head pipeline.

Convention: this package exposes a `build(config)` factory that returns
`(LightningModule, train_loader, val_loader)`. The shared
`pato.pipelines.train.train()` calls `build(config)` via importlib —
it never imports anything from this package directly.
"""

from pato.pipelines.sam_head.config import SAMHeadRunConfig


def build(config: SAMHeadRunConfig):
    from pato.pipelines.sam_head.data import make_dataloaders
    from pato.pipelines.sam_head.module import SAMHeadLightning

    train_loader, val_loader = make_dataloaders(config)
    kwargs = dict(
        num_classes=config.num_classes,
        feature_channels=config.feature_channels,
        head_widths=tuple(config.head_widths),
        learning_rate=config.learning_rate,
    )
    if config.init_from_checkpoint is not None:
        model = SAMHeadLightning.load_from_checkpoint(config.init_from_checkpoint, **kwargs)
    else:
        model = SAMHeadLightning(**kwargs)
    return model, train_loader, val_loader
