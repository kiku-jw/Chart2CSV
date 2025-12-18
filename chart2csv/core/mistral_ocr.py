"""
Mistral OCR backend for Chart2CSV.

This module handles interaction with Mistral's OCR API.
"""

import os
import base64
import re
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

def get_mistral_client() -> Optional[Any]:
    """Get Mistral client if API key is present."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return None

    if not MISTRAL_AVAILABLE:
        return None

    return Mistral(api_key=api_key)

def encode_image_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string."""
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def extract_numbers_from_mistral(image: np.ndarray) -> List[float]:
    """
    Extract numbers from an image strip using Mistral OCR.

    Args:
        image: Image strip (e.g. X-axis or Y-axis region)

    Returns:
        List of parsed numbers found in the image.
    """
    client = get_mistral_client()
    if not client:
        # Fallback or error if client not available
        # In a real scenario, we might raise an error, but for "basis" we assume key exists.
        print("Warning: MISTRAL_API_KEY not set or mistralai not installed.")
        return []

    # Prepare document payload
    base64_image = encode_image_base64(image)

    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": base64_image
            },
            include_image_base64=False
        )

        # Parse markdown from response
        # The response structure: ocr_response.pages[0].markdown
        # We assume single page for an image
        markdown = ocr_response.pages[0].markdown

        # Extract numbers from markdown
        # Mistral might return them as a list, or just text.
        # We look for number patterns.
        # This is a heuristic.
        numbers = parse_numbers_from_text(markdown)
        return numbers

    except Exception as e:
        print(f"Mistral OCR failed: {e}")
        return []

def parse_numbers_from_text(text: str) -> List[float]:
    """
    Parse all numbers from a text string.
    """
    # Regex for numbers (integers, floats, scientific notation)
    # Avoid picking up things that look like dates or indices if possible
    # But for axis labels, they are usually just numbers.
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    matches = re.findall(pattern, text)

    values = []
    for match in matches:
        try:
            # Filter out single dots or empty strings if regex is loose
            if match in ['.', '', '-', '+']:
                continue
            values.append(float(match))
        except ValueError:
            continue

    return values

class MistralOCRBackend:
    """Wrapper for Mistral OCR operations."""

    def __init__(self):
        self.client = get_mistral_client()

    def is_available(self) -> bool:
        return self.client is not None

    def process_axis_strip(self, image: np.ndarray) -> List[float]:
        """Process an axis strip image and return detected numbers."""
        if not self.is_available():
            return []
        return extract_numbers_from_mistral(image)
