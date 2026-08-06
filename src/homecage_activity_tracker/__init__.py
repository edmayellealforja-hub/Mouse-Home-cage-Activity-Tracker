"""Quality-aware ROI occupancy analysis for multi-animal DLC data."""

from .analysis import AnalysisResult, analyze_occupancy
from .dlc import DLCData, load_dlc
from .rois import ROIConfig, load_rois

__all__ = [
    "AnalysisResult",
    "DLCData",
    "ROIConfig",
    "analyze_occupancy",
    "load_dlc",
    "load_rois",
]

__version__ = "0.1.0"
