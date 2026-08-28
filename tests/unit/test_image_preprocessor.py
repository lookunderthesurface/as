from __future__ import annotations

import unittest
from pathlib import Path

from secretary.inference.image import ImagePreprocessor


class ImagePreprocessorTests(unittest.TestCase):
    FIXTURES = Path(__file__).parents[1] / "fixtures"

    def test_landscape_preserves_aspect_ratio_and_caps_long_edge(self) -> None:
        result = ImagePreprocessor(max_long_edge=4).prepare_image(self.FIXTURES / "landscape.ppm")

        self.assertIsNotNone(result)
        self.assertEqual((result.width, result.height), (4, 2))
        self.assertIn(result.mime_type, {"image/jpeg", "image/png"})
        self.assertTrue(result.data)

    def test_portrait_preserves_aspect_ratio(self) -> None:
        result = ImagePreprocessor(max_long_edge=4).prepare_image(self.FIXTURES / "portrait.ppm")

        self.assertIsNotNone(result)
        self.assertEqual((result.width, result.height), (2, 4))

    def test_small_image_is_not_upscaled(self) -> None:
        result = ImagePreprocessor(max_long_edge=1280).prepare_image(self.FIXTURES / "small.ppm")

        self.assertIsNotNone(result)
        self.assertEqual((result.width, result.height), (2, 1))

    def test_large_image_is_resized(self) -> None:
        result = ImagePreprocessor(max_long_edge=4).prepare_image(self.FIXTURES / "large.ppm")

        self.assertIsNotNone(result)
        self.assertEqual((result.width, result.height), (4, 2))

    def test_missing_and_corrupt_images_fail_safely(self) -> None:
        preprocessor = ImagePreprocessor()

        self.assertIsNone(preprocessor.prepare_image(self.FIXTURES / "missing.ppm"))
        self.assertIsNone(preprocessor.prepare_image(self.FIXTURES / "corrupt_image.ppm"))
