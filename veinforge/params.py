from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass
class Params:
    pixel_size_um: float | None = None          # microns per pixel; None -> pixel units + warn
    channel: str = "gray"                        # "gray" | "r" | "g" | "b"
    invert: bool = True                          # make veins bright after preprocessing
    background_radius: int = 50                  # rolling-ball radius (px); 0 disables
    clahe_clip: float = 0.01                     # 0 disables CLAHE
    sato_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    threshold_method: str = "otsu"               # "otsu" | "hysteresis"
    hysteresis_low: float = 0.1                  # used when threshold_method == "hysteresis"
    hysteresis_high: float = 0.25
    min_object_px: int = 64                      # remove specks smaller than this
    closing_radius: int = 1                      # morphological closing radius (px)
    filename_pattern: str = (
        r"(?P<sample_id>[^_]+)_(?P<treatment>[^_]+)_(?P<replicate>[^_]+)_(?P<position>[^_.]+)"
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Params":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if data.get("sato_sigmas") is not None:
            data["sato_sigmas"] = tuple(data["sato_sigmas"])
        return cls(**data)
