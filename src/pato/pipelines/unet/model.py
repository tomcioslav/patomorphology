from monai.networks.nets import UNet


def build_unet(
    num_classes: int = 2,
    in_channels: int = 3,
    channels: tuple[int, ...] = (32, 64, 128, 256, 512),
    num_res_units: int = 2,
) -> UNet:
    """Build a 2D MONAI U-Net configured for histopathology segmentation.

    `strides` are always 2 between every consecutive pair of `channels`
    (standard U-Net halving). To change the depth, change `channels`.
    """
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=channels,
        strides=(2,) * (len(channels) - 1),
        num_res_units=num_res_units,
    )
