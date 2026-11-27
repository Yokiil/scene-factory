"""XML-to-3D spatial geometry, analytical collision checking, S-Space benchmark, and readout math."""

from .benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkSample,
    generate_xml_spatial_benchmark,
)
from .geometry import (
    BodyGeometry,
    GeomOBB,
    PairCollision,
    SpatialAnalysis,
    analyze_xml_spatial,
    obb_signed_distance,
)
from .sspace_probe import (
    SpatialBoxProbe,
    cross_modal_subspace_alignment,
    evaluate_collision_methods,
    fit_contrast_axis,
    fit_spatial_box_probe,
    predict_box_collision,
    steering_delta_for_separation,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkSample",
    "BodyGeometry",
    "GeomOBB",
    "PairCollision",
    "SpatialAnalysis",
    "SpatialBoxProbe",
    "analyze_xml_spatial",
    "cross_modal_subspace_alignment",
    "evaluate_collision_methods",
    "fit_contrast_axis",
    "fit_spatial_box_probe",
    "generate_xml_spatial_benchmark",
    "obb_signed_distance",
    "predict_box_collision",
    "steering_delta_for_separation",
]
