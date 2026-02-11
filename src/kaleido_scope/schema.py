from enum import Enum

from pydantic import Field
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class MirrorMode(str, Enum):
    NONE = "none"
    MIRROR_2X = "2x"
    MIRROR_4X = "4x"
    KALEIDO_6 = "kaleido6"  # <-- NEW


class KaleidoScopeConfig(BasePipelineConfig):
    """
    Main pipeline configuration for Kaleido Scope.
    This version does not set `usage`, so it appears in the main pipeline selector.
    """

    pipeline_id = "kaleido-scope"
    pipeline_name = "Kaleido Scope"
    pipeline_description = "GPU kaleidoscope/mirror: 2x/4x, N-fold symmetry, rotation, zoom, warp"
    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}

    enabled: bool = Field(
        default=True,
        description="Enable the effect (off returns the original video)",
        json_schema_extra=ui_field_config(order=1, label="Enabled", category="configuration"),
    )

    mix: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend between original (0) and fully effected (1)",
        json_schema_extra=ui_field_config(order=2, label="Mix", category="input"),
    )

    # --- Mirror / Quick Modes ---
    mirror_mode: MirrorMode = Field(
        default=MirrorMode.NONE,
        description="Mirror symmetry mode: none, 2-way, 4-way, or a quick 6-slice kaleido preset",
        json_schema_extra=ui_field_config(order=10, label="Mode", category="configuration"),
    )

    # --- Rotational kaleidoscope ---
    rotational_enabled: bool = Field(
        default=True,
        description="Enable N-fold rotational symmetry (kaleidoscope folding)",
        json_schema_extra=ui_field_config(order=20, label="Rotational Symmetry", category="configuration"),
    )

    rotational_slices: int = Field(
        default=6,
        ge=3,
        le=12,
        description="Number of symmetry slices (N). Higher = more segments",
        json_schema_extra=ui_field_config(order=21, label="Slices (N)", category="configuration"),
    )

    rotation_deg: float = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Rotate the kaleidoscope pattern (degrees)",
        json_schema_extra=ui_field_config(order=22, label="Rotation", category="input"),
    )

    # --- Optional spatial tweaks ---
    zoom: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Zoom into the source before applying symmetry (1 = no zoom)",
        json_schema_extra=ui_field_config(order=30, label="Zoom", category="configuration"),
    )

    warp: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Radial warp amount (0 = off). Positive expands edges; negative compresses",
        json_schema_extra=ui_field_config(order=31, label="Warp", category="configuration"),
    )


class KaleidoScopePreConfig(KaleidoScopeConfig):
    """Preprocessor variant (appears in the Preprocessor dropdown)."""

    pipeline_id = "kaleido-scope-pre"
    pipeline_name = "Kaleido Scope (Pre)"
    pipeline_description = "Preprocess input video with kaleidoscope/mirror symmetry"
    usage = [UsageType.PREPROCESSOR]


class KaleidoScopePostConfig(KaleidoScopeConfig):
    """Post-processor variant (appears in the Post-processor slot)."""

    pipeline_id = "kaleido-scope-post"
    pipeline_name = "Kaleido Scope (Post)"
    pipeline_description = "Post-process output video with kaleidoscope/mirror symmetry"
    usage = [UsageType.POSTPROCESSOR]
