"""
Core data types for Chart2CSV.

This module defines the data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


# Confidence weights for overall score calculation
# These weights reflect the relative importance of each component
CONFIDENCE_WEIGHTS = {
    'crop': 0.3,       # Crop quality is critical - affects all downstream processing
    'axis': 0.25,      # Axis detection enables coordinate transformation
    'ocr': 0.3,        # OCR accuracy determines scale precision
    'extraction': 0.15  # Extraction confidence is relative to prior steps
}


class ChartType(Enum):
    """Supported chart types."""
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    UNKNOWN = "unknown"


class Scale(Enum):
    """Axis scale types."""
    LINEAR = "linear"
    LOG = "log"


class WarningCode(Enum):
    """Warning codes for quality issues."""
    LOW_RESOLUTION = "LOW_RESOLUTION"
    CROP_UNCERTAIN = "CROP_UNCERTAIN"
    AXES_UNCERTAIN = "AXES_UNCERTAIN"
    SKEW_DETECTED = "SKEW_DETECTED"
    OCR_FAILED = "OCR_FAILED"
    OCR_PARTIAL = "OCR_PARTIAL"
    POSSIBLE_LOG_SCALE = "POSSIBLE_LOG_SCALE"
    MULTI_SERIES_DETECTED = "MULTI_SERIES_DETECTED"
    LEGEND_DETECTED = "LEGEND_DETECTED"
    NOISE_DETECTED = "NOISE_DETECTED"
    LINE_GAPS = "LINE_GAPS"
    FEW_POINTS = "FEW_POINTS"
    ROTATION_DETECTED = "ROTATION_DETECTED"


@dataclass
class Warning:
    """A quality warning with code and recommendation."""
    code: WarningCode
    message: str
    recommendation: str


@dataclass
class AxisInfo:
    """Information about a detected axis."""
    min_value: float
    max_value: float
    scale: Scale
    pixel_start: int
    pixel_end: int
    ticks_detected: int
    ticks_parsed: int
    ocr_confidence: float


@dataclass
class Confidence:
    """Detailed confidence breakdown."""
    crop: float  # 0.0-1.0
    axis: float  # 0.0-1.0
    ocr: float   # 0.0-1.0
    extraction: float  # 0.0-1.0

    def overall(self) -> float:
        """
        Calculate overall confidence as weighted average.

        Uses CONFIDENCE_WEIGHTS constants for reproducibility.

        Formula:
            confidence = (
                0.3 × crop +
                0.25 × axis +
                0.3 × ocr +
                0.15 × extraction
            )
        """
        return (
            CONFIDENCE_WEIGHTS['crop'] * self.crop +
            CONFIDENCE_WEIGHTS['axis'] * self.axis +
            CONFIDENCE_WEIGHTS['ocr'] * self.ocr +
            CONFIDENCE_WEIGHTS['extraction'] * self.extraction
        )

    def zone(self) -> str:
        """Return confidence zone: high, medium, or low."""
        overall = self.overall()
        if overall >= 0.7:
            return "high"
        elif overall >= 0.4:
            return "medium"
        else:
            return "low"


@dataclass
class ChartResult:
    """Complete result of chart extraction."""

    # Chart information
    chart_type: ChartType

    # Extracted data (Nx2 array for scatter/line, Nx1 for bar)
    data: np.ndarray

    # Axis information
    x_axis: AxisInfo
    y_axis: AxisInfo

    # Quality metrics
    confidence: Confidence
    warnings: List[Warning] = field(default_factory=list)

    # Metadata
    num_points: int = 0
    runtime_ms: float = 0.0

    # Extraction parameters (for reproducibility)
    extraction_params: Dict[str, Any] = field(default_factory=dict)

    # Original image path
    image_path: Optional[str] = None

    # Overlay image (if generated)
    overlay: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "chart_type": self.chart_type.value,
            "confidence": self.confidence.overall(),
            "confidence_breakdown": {
                "crop": self.confidence.crop,
                "axis": self.confidence.axis,
                "ocr": self.confidence.ocr,
                "extraction": self.confidence.extraction,
                "zone": self.confidence.zone()
            },
            "warnings": [
                {
                    "code": w.code.value,
                    "message": w.message,
                    "recommendation": w.recommendation
                }
                for w in self.warnings
            ],
            "axes": {
                "x": {
                    "min": float(self.x_axis.min_value),
                    "max": float(self.x_axis.max_value),
                    "scale": self.x_axis.scale.value,
                    "ticks_detected": self.x_axis.ticks_detected,
                    "ticks_parsed": self.x_axis.ticks_parsed,
                    "ocr_confidence": self.x_axis.ocr_confidence
                },
                "y": {
                    "min": float(self.y_axis.min_value),
                    "max": float(self.y_axis.max_value),
                    "scale": self.y_axis.scale.value,
                    "ticks_detected": self.y_axis.ticks_detected,
                    "ticks_parsed": self.y_axis.ticks_parsed,
                    "ocr_confidence": self.y_axis.ocr_confidence
                }
            },
            "num_points": self.num_points,
            "runtime_ms": self.runtime_ms,
            "extraction_params": self.extraction_params,
            "image_path": self.image_path
        }

    def add_warning(self, code: WarningCode, message: str, recommendation: str):
        """Add a warning to the result."""
        self.warnings.append(Warning(
            code=code,
            message=message,
            recommendation=recommendation
        ))


@dataclass
class CropBox:
    """Bounding box for plot area crop."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    method: str  # "auto" or "manual"

    def width(self) -> int:
        return self.x2 - self.x1

    def height(self) -> int:
        return self.y2 - self.y1

    def area(self) -> int:
        return self.width() * self.height()


@dataclass
class AxisLine:
    """Detected axis line."""
    pixel_position: int  # X position for Y-axis, Y position for X-axis
    orientation: str  # "vertical" or "horizontal"
    confidence: float
    method: str  # "hough", "projection", "manual"


@dataclass
class Tick:
    """A single axis tick."""
    pixel_position: int
    value: Optional[float] = None
    ocr_text: Optional[str] = None
    confidence: float = 0.0
    parsed: bool = False
