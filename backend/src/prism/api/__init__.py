"""Programmatic research API and service layer for PRISM."""

from prism.api.geometry_service import (
    GeometryService,
    ObservatoryExperimentMeta,
    generate_observatory_demo_data,
)
from prism.api.robustness_service import (
    RobustnessExperimentMeta,
    RobustnessService,
    generate_robustness_demo_data,
)

__all__ = [
    "GeometryService",
    "ObservatoryExperimentMeta",
    "RobustnessExperimentMeta",
    "RobustnessService",
    "generate_observatory_demo_data",
    "generate_robustness_demo_data",
]
