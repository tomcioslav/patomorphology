from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseSettings):

    root: Path = Path(__file__).resolve().parent
    data: Path = root / "data"
    data_raw: Path = root / "data" / "raw"

    nmsc: Path = data_raw / "nmsc-segmentation" / "data"
    nmsc_1x: Path = nmsc / "1x"
    nmsc_2x: Path = nmsc / "2x"
    nmsc_5x: Path = nmsc / "5x"
    nmsc_10x: Path = nmsc / "10x"
    nmsc_margin: Path = nmsc / "MarginData"


paths = Paths()
