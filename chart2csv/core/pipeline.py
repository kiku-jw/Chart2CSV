"""
Main extraction pipeline for Chart2CSV.

This is the entry point for chart data extraction.
"""

import time
from pathlib import Path
from typing import Union, Optional, Dict, Any
import numpy as np
import cv2

from chart2csv.core.types import (
    ChartResult,
    ChartType,
    Confidence,
    AxisInfo,
    Scale,
    WarningCode
)
from chart2csv.core.preprocess import preprocess_image, detect_plot_area
from chart2csv.core.detection import detect_axes, detect_ticks
from chart2csv.core.ocr import extract_tick_labels
from chart2csv.core.transform import build_transform, apply_transform
from chart2csv.core.export import generate_overlay
from chart2csv.core.extraction import (
    extract_scatter_points,
    extract_line_points,
    extract_bar_data
)

def extract_chart(
    image_path: Union[str, Path],
    crop: Optional[tuple[int, int, int, int]] = None,
    x_axis_pos: Optional[int] = None,
    y_axis_pos: Optional[int] = None,
    x_scale: Scale = Scale.LINEAR,
    y_scale: Scale = Scale.LINEAR,
    chart_type: Optional[ChartType] = None,
    calibration_points: Optional[Dict[str, Any]] = None,
    generate_overlay_image: bool = True,
    use_mistral: bool = False
) -> ChartResult:
    """
    Extract data from a chart image.
    """
    start_time = time.time()
    image_path = str(image_path)

    # Initialize extraction parameters
    params = {
        "crop": "manual" if crop else "auto",
        "x_axis": "manual" if x_axis_pos else "auto",
        "y_axis": "manual" if y_axis_pos else "auto",
        "x_scale": x_scale.value,
        "y_scale": y_scale.value,
        "chart_type": chart_type.value if chart_type else "auto",
        "calibration": "manual" if calibration_points else "auto"
    }

    # Load image robustly using PIL (OpenCV imread can fail in some environments)
    try:
        from PIL import Image
        pil_img = Image.open(image_path)
        # Convert to BGR for OpenCV compatibility
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        image = np.array(pil_img)
        # RGB to BGR
        image = image[:, :, ::-1].copy()
    except Exception as e:
        # Fallback to cv2.imread if PIL fails
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path} (Error: {e})")

    # Initialize confidence components
    crop_conf = 0.0
    axis_conf = 0.0
    ocr_conf = 0.0
    extraction_conf = 0.0
    warnings_list = []

    # Step 1: Preprocess image
    processed = preprocess_image(image)

    # Step 2: Detect plot area (or use manual crop)
    if crop:
        x1, y1, x2, y2 = crop
        crop_box = (x1, y1, x2, y2)
        crop_conf = 1.0
    else:
        crop_box, crop_conf = detect_plot_area(image)  # Use original for better detection
        if crop_box is None:
            crop_box = (0, 0, image.shape[1], image.shape[0])
            crop_conf = 0.1
            
        if crop_conf < 0.6:
            warnings_list.append((
                WarningCode.CROP_UNCERTAIN,
                f"Auto crop confidence low: {crop_conf:.2f}",
                "Check overlay or use --crop x1,y1,x2,y2"
            ))

    # Crop to plot area
    x1, y1, x2, y2 = crop_box
    cropped = processed[y1:y2, x1:x2]

    # Check resolution
    h, w = cropped.shape[:2]
    if max(h, w) < 600:
        warnings_list.append((
            WarningCode.LOW_RESOLUTION,
            f"Image resolution low: {w}x{h}",
            "Use higher resolution scan/screenshot"
        ))

    # Step 3: Detect axes (or use manual positions)
    if x_axis_pos is not None and y_axis_pos is not None:
        axes = {"x": x_axis_pos, "y": y_axis_pos}
        axis_conf = 1.0
    else:
        axes, axis_conf = detect_axes(processed) # Use full processed image
        if axis_conf < 0.5:
            warnings_list.append((
                WarningCode.AXES_UNCERTAIN,
                f"Axis detection confidence low: {axis_conf:.2f}",
                "Use --x-axis y=PX --y-axis x=PX"
            ))

    # Step 4: Detect and OCR ticks
    ticks = None
    if calibration_points:
        ocr_conf = 1.0
    else:
        ticks, ocr_conf = extract_tick_labels(processed, axes, use_mistral=use_mistral)
        if ocr_conf < 0.4:
            warnings_list.append((
                WarningCode.OCR_FAILED,
                f"OCR failed: {ocr_conf:.1%} success rate",
                "Use --calibrate (manual input)"
            ))

    # Step 5: Build pixel→value transform
    transform = None
    if calibration_points:
        transform, fit_error = build_transform(
            calibration_points=calibration_points,
            x_scale=x_scale,
            y_scale=y_scale
        )
    elif ticks and any(ticks.values()):
        transform, fit_error = build_transform(
            ticks=ticks,
            x_scale=x_scale,
            y_scale=y_scale
        )
        if fit_error > 0.1 and x_scale == Scale.LINEAR and y_scale == Scale.LINEAR:
            warnings_list.append((
                WarningCode.POSSIBLE_LOG_SCALE,
                f"Linear fit error high: {fit_error:.1%}",
                "Use --calibrate --y-scale log"
            ))
    else:
        # Emergency fallback if no ticks found and no calibration
        transform = {
            "x": {"a": 1.0, "b": 0.0, "scale": "linear"},
            "y": {"a": 1.0, "b": 0.0, "scale": "linear"}
        }
        fit_error = 1.0

    # Step 6: Detect chart type (if not manual)
    if chart_type is None or chart_type == ChartType.UNKNOWN:
        chart_type = detect_chart_type(cropped)

    # Step 7: Extract data based on chart type
    if chart_type == ChartType.SCATTER:
        points_px, extraction_conf = extract_scatter_points(image, crop_box)
    elif chart_type == ChartType.LINE:
        points_px, extraction_conf = extract_line_points(image, crop_box)
    elif chart_type == ChartType.BAR:
        points_px, extraction_conf = extract_bar_data(image, crop_box)
    else:
        # Fallback to scatter
        points_px, extraction_conf = extract_scatter_points(image, crop_box)

    # Ensure 2D array (N, 2) even if empty
    points_px = np.array(points_px).reshape(-1, 2)
    data = apply_transform(points_px, transform)

    # Step 8: Generate overlay (if requested)
    overlay_img = None
    if generate_overlay_image:
        overlay_img = generate_overlay(image, points_px, crop_box, axes, chart_type)

    # Build result
    confidence = Confidence(
        crop=crop_conf,
        axis=axis_conf,
        ocr=ocr_conf,
        extraction=extraction_conf
    )

    # Fill AxisInfo from transform and ticks
    x_axis_info = AxisInfo(
        min_value=transform["x"]["b"],
        max_value=transform["x"]["a"] * w + transform["x"]["b"],
        scale=x_scale,
        pixel_start=axes["y"],
        pixel_end=w,
        ticks_detected=len(ticks["x"]) if ticks else 0,
        ticks_parsed=len([p for p in ticks["x"] if p.get("value") is not None]) if ticks else 0,
        ocr_confidence=ocr_conf
    )

    y_axis_info = AxisInfo(
        min_value=transform["y"]["b"],
        max_value=transform["y"]["a"] * h + transform["y"]["b"],
        scale=y_scale,
        pixel_start=axes["x"],
        pixel_end=0,
        ticks_detected=len(ticks["y"]) if ticks else 0,
        ticks_parsed=len([p for p in ticks["y"] if p.get("value") is not None]) if ticks else 0,
        ocr_confidence=ocr_conf
    )

    result = ChartResult(
        chart_type=chart_type,
        data=data,
        x_axis=x_axis_info,
        y_axis=y_axis_info,
        confidence=confidence,
        num_points=len(data),
        runtime_ms=(time.time() - start_time) * 1000,
        extraction_params=params,
        image_path=image_path,
        overlay=overlay_img
    )

    for code, message, recommendation in warnings_list:
        result.add_warning(code, message, recommendation)

    return result


def detect_chart_type(image: np.ndarray) -> ChartType:
    """
    Detect chart type from image features.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Simple heuristic based on contour count and aspect ratios
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return ChartType.SCATTER
        
    num_contours = len(contours)
    bar_like = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 2*w and h > 20: bar_like += 1
        
    if bar_like > 3 and bar_like > num_contours * 0.5:
        return ChartType.BAR
        
    # Check for line-like (one large continuous contour)
    if num_contours < 5 and any(cv2.arcLength(c, False) > 200 for c in contours):
        return ChartType.LINE
        
    return ChartType.SCATTER

