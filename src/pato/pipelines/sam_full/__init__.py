"""SAM-full pipeline.

End-to-end fine-tuning: SAM encoder + decoder head, both trainable.
Trains on raw image tiles from a normalized dataset (no feature cache).
Exposes `build(config)` for the shared `pato.pipelines.train.train()`.
"""

from pato.pipelines.sam_full.config import SAMFullRunConfig


def build(config: SAMFullRunConfig):
    from pato.pipelines.sam_full.data import make_dataloaders
    from pato.pipelines.sam_full.module import SAMFullLightning

    train_loader, val_loader = make_dataloaders(config)
    kwargs = dict(
        sam_model=config.sam_model,
        num_classes=config.num_classes,
        feature_channels=config.feature_channels,
        head_widths=tuple(config.head_widths),
        learning_rate=config.learning_rate,
        sam_learning_rate=config.sam_learning_rate,
    )
    if config.init_from_checkpoint is not None:
        model = SAMFullLightning.load_from_checkpoint(config.init_from_checkpoint, **kwargs)
    else:
        model = SAMFullLightning(**kwargs)
    return model, train_loader, val_loader
