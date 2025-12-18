"""
Axis and tick detection for Chart2CSV.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, List


def detect_axes(image: np.ndarray) -> Tuple[Dict[str, int], float]:
    """
    Detect X and Y axes in the chart image.

    Uses Hough line detection to find two dominant perpendicular lines.
    Falls back to projection-based method if Hough fails.

    Args:
        image: Cropped and preprocessed chart image

    Returns:
        Tuple of (axes_dict, confidence)
        axes_dict: {"x": y_position, "y": x_position}
        confidence: 0.0-1.0
    """
    h, w = image.shape[:2]
    
    # Edge detection for line finding
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Hough line transform
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=min(w, h) // 4,
        maxLineGap=20
    )
    
    horizontals = []
    verticals = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
            
            # Normalize angle to 0-180
            if angle > 90:
                angle = 180 - angle
                
            if angle < 5:  # Horizontal
                horizontals.append({"pos": (y1 + y2) // 2, "len": length})
            elif 85 < angle < 95:  # Vertical
                verticals.append({"pos": (x1 + x2) // 2, "len": length})

    # Find the most likely axes:
    # X-axis is usually the lowest long horizontal line
    # Y-axis is usually the leftmost long vertical line
    
    x_axis_y = h - 1
    y_axis_x = 0
    
    axis_conf = 0.0
    
    if horizontals:
        # Sort by position (bottom first), then length
        horizontals.sort(key=lambda l: (-l["pos"], -l["len"]))
        x_axis_y = horizontals[0]["pos"]
        axis_conf += 0.4
        
    if verticals:
        # Sort by position (left first), then length
        verticals.sort(key=lambda l: (l["pos"], -l["len"]))
        y_axis_x = verticals[0]["pos"]
        axis_conf += 0.4

    # Fallback to projection if missing or low confidence
    if not horizontals or not verticals:
        # Simple projection: find where dark pixels are concentrated at edges
        row_sum = np.sum(255 - image, axis=1)
        col_sum = np.sum(255 - image, axis=0)
        
        if not horizontals:
            x_axis_y = np.argmax(row_sum[h//2:]) + h//2
        if not verticals:
            y_axis_x = np.argmax(col_sum[:w//2])
            
        axis_conf = max(axis_conf, 0.4)

    # Check for perpendicularity/intersection
    if horizontals and verticals:
        axis_conf = min(1.0, axis_conf + 0.2)

    return {"x": int(x_axis_y), "y": int(y_axis_x)}, axis_conf


def detect_ticks(
    image: np.ndarray,
    axes: Dict[str, int]
) -> Tuple[Dict[str, List[int]], float]:
    """
    Detect tick mark positions along axes.

    Args:
        image: Cropped chart image
        axes: Axis positions from detect_axes()

    Returns:
        Tuple of (ticks_dict, confidence)
        ticks_dict: {"x": [x_positions], "y": [y_positions]}
    """
    h, w = image.shape[:2]
    x_axis_y = axes["x"]
    y_axis_x = axes["y"]
    
    ticks = {"x": [], "y": []}
    
    # X-axis ticks (vertical marks below or on the axis)
    # Target region: a strip around the x-axis
    y_min = max(0, x_axis_y - 10)
    y_max = min(h, x_axis_y + 15)
    x_strip = image[y_min:y_max, :]
    
    # Edge detection on strip
    x_edges = cv2.Sobel(x_strip, cv2.CV_8U, 1, 0, ksize=3)
    x_proj = np.sum(x_edges, axis=0)
    
    # Ensure 1D array
    if x_proj.ndim > 1:
        x_proj = x_proj.flatten()

    # Find peaks in projection
    from scipy.signal import find_peaks
    x_peaks, _ = find_peaks(x_proj, distance=10, height=np.mean(x_proj))
    ticks["x"] = [int(p) for p in x_peaks]
    
    # Y-axis ticks (horizontal marks to the left of the axis)
    x_min = max(0, y_axis_x - 15)
    x_max = min(w, y_axis_x + 10)
    y_strip = image[:, x_min:x_max]
    
    y_edges = cv2.Sobel(y_strip, cv2.CV_8U, 0, 1, ksize=3)
    y_proj = np.sum(y_edges, axis=1)
    
    # Ensure 1D array
    if y_proj.ndim > 1:
        y_proj = y_proj.flatten()

    y_peaks, _ = find_peaks(y_proj, distance=10, height=np.mean(y_proj))
    ticks["y"] = [int(p) for p in y_peaks]
    
    # Calculate confidence based on number of ticks and regularity
    conf = 0.0
    if len(ticks["x"]) >= 2: conf += 0.4
    if len(ticks["y"]) >= 2: conf += 0.4
    
    # Check for regularity in spacing
    if len(ticks["x"]) >= 3:
        diffs = np.diff(ticks["x"])
        if np.std(diffs) / np.mean(diffs) < 0.2:
            conf += 0.1
            
    if len(ticks["y"]) >= 3:
        diffs = np.diff(ticks["y"])
        if np.std(diffs) / np.mean(diffs) < 0.2:
            conf += 0.1

    return ticks, min(1.0, conf)
