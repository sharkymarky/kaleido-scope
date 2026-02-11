from typing import TYPE_CHECKING

import torch
from scope.core.pipelines.interface import Pipeline, Requirements

from .effects.kaleido import kaleido_effect
from .schema import KaleidoScopeConfig, KaleidoScopePreConfig, KaleidoScopePostConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class _BaseKaleidoPipeline(Pipeline):
    CONFIG_CLASS: type["BasePipelineConfig"] = KaleidoScopeConfig

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return cls.CONFIG_CLASS

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("kaleido-scope pipelines require video input")

        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        out = kaleido_effect(
            frames=frames,
            enabled=kwargs.get("enabled", True),
            mix=kwargs.get("mix", 1.0),
            mirror_mode=kwargs.get("mirror_mode", "none"),
            rotational_enabled=kwargs.get("rotational_enabled", True),
            rotational_slices=kwargs.get("rotational_slices", 6),
            rotation_deg=kwargs.get("rotation_deg", 0.0),
            zoom=kwargs.get("zoom", 1.0),
            warp=kwargs.get("warp", 0.0),
        )
        return {"video": out.clamp(0, 1)}


class KaleidoScopePipeline(_BaseKaleidoPipeline):
    CONFIG_CLASS = KaleidoScopeConfig


class KaleidoScopePrePipeline(_BaseKaleidoPipeline):
    CONFIG_CLASS = KaleidoScopePreConfig


class KaleidoScopePostPipeline(_BaseKaleidoPipeline):
    CONFIG_CLASS = KaleidoScopePostConfig
