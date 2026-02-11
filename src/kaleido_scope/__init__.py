"""
kaleido-scope: a GPU-accelerated kaleidoscope / mirror effect plugin for Scope.

Entry point declared in pyproject.toml under:
[project.entry-points."scope"]
kaleido_scope = "kaleido_scope"
"""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """
    Called by Scope when loading this plugin.
    We register three pipelines:
      - Kaleido Scope (main pipeline)
      - Kaleido Scope (Pre) (preprocessor)
      - Kaleido Scope (Post) (post-processor)
    """
    # Lazy import: avoid importing torch/effect code until Scope loads the plugin.
    from .pipeline import (
        KaleidoScopePipeline,
        KaleidoScopePrePipeline,
        KaleidoScopePostPipeline,
    )

    register(KaleidoScopePipeline)
    register(KaleidoScopePrePipeline)
    register(KaleidoScopePostPipeline)
