
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mocking mistralai if not installed or for testing
import sys
sys.modules['mistralai'] = MagicMock()

from chart2csv.core.mistral_ocr import MistralOCRBackend, extract_numbers_from_mistral, parse_numbers_from_text
from chart2csv.core.ocr import extract_tick_labels

class TestMistralOCR(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_ocr = MagicMock()
        self.mock_client.ocr = self.mock_ocr

        # Mock response structure
        self.mock_response = MagicMock()
        self.mock_page = MagicMock()
        self.mock_response.pages = [self.mock_page]

    @patch('chart2csv.core.mistral_ocr.get_mistral_client')
    def test_extract_numbers(self, mock_get_client):
        mock_get_client.return_value = self.mock_client
        self.mock_page.markdown = "Values are 10, 20.5, and 3e2."
        self.mock_ocr.process.return_value = self.mock_response

        # Create dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        numbers = extract_numbers_from_mistral(img)
        self.assertEqual(numbers, [10.0, 20.5, 300.0])

    def test_parse_numbers(self):
        text = "X axis: 0  10  20  30  40"
        nums = parse_numbers_from_text(text)
        self.assertEqual(nums, [0, 10, 20, 30, 40])

        text_complex = "Log scale: 1e-1, 1e0, 1e1"
        nums = parse_numbers_from_text(text_complex)
        self.assertEqual(nums, [0.1, 1.0, 10.0])

    @patch('chart2csv.core.ocr.MistralOCRBackend')
    @patch('chart2csv.core.detection.detect_ticks')
    def test_extract_tick_labels_mistral(self, mock_detect, MockBackend):
        # Setup mock backend
        backend_instance = MockBackend.return_value
        backend_instance.is_available.return_value = True
        backend_instance.process_axis_strip.side_effect = [
            [10, 20, 30], # X axis
            [5, 10]       # Y axis
        ]

        # Setup detect ticks
        mock_detect.return_value = (
            {"x": [100, 200, 300], "y": [50, 100]}, # 3 X ticks, 2 Y ticks
            0.9
        )

        # Create dummy image
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        axes = {"x": 400, "y": 50}

        ticks_data, conf = extract_tick_labels(img, axes, use_mistral=True)

        # Verify calls
        self.assertTrue(backend_instance.process_axis_strip.called)

        # Verify results
        self.assertEqual(len(ticks_data["x"]), 3)
        self.assertEqual(ticks_data["x"][0]["value"], 10)
        self.assertEqual(ticks_data["x"][1]["value"], 20)
        self.assertEqual(ticks_data["x"][2]["value"], 30)

        self.assertEqual(len(ticks_data["y"]), 2)
        # Y axis sorting: ticks["y"] usually sorted by pixel (small to large)
        # Mistral returns [5, 10].
        # Ticks detected at 50, 100.
        # 50 -> 5, 100 -> 10.
        self.assertEqual(ticks_data["y"][0]["value"], 5)
        self.assertEqual(ticks_data["y"][1]["value"], 10)

if __name__ == '__main__':
    unittest.main()
