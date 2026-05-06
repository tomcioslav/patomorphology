from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data: Path = Field(default=Path("data"))
    notebooks: Path = Field(default=Path("notebooks"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PATO_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    paths: PathsConfig = PathsConfig()
    seed: int = 42
