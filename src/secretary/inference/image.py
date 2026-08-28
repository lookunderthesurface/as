from __future__ import annotations

import base64
import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EncodedImage:
    data: str
    mime_type: str
    width: int
    height: int


class ImagePreprocessor:
    """Resize and encode images entirely in memory with safe failure semantics."""

    def __init__(self, max_long_edge: int = 1280, jpeg_quality: int = 85) -> None:
        self.max_long_edge = max(1, max_long_edge)
        self.jpeg_quality = min(95, max(1, jpeg_quality))

    def prepare_image(self, path: str | Path) -> EncodedImage | None:
        image_path = Path(path)
        try:
            if not image_path.is_file():
                return None
            prepared = self._prepare_with_pillow(image_path)
            if prepared is not None:
                return prepared
            pixels, width, height = self._read_ppm(image_path.read_bytes())
            resized, new_width, new_height = self._resize_rgb(pixels, width, height)
            encoded = self._encode_png(resized, new_width, new_height)
            return EncodedImage(
                data=base64.b64encode(encoded).decode("ascii"),
                mime_type="image/png",
                width=new_width,
                height=new_height,
            )
        except (OSError, ValueError, TypeError, OverflowError, zlib.error):
            # Image input is optional context; it must never take down Secretary.
            return None

    def _prepare_with_pillow(self, path: Path) -> EncodedImage | None:
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError:
            return None
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((self.max_long_edge, self.max_long_edge), getattr(Image, "Resampling", Image).LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=self.jpeg_quality, optimize=True)
            return EncodedImage(
                data=base64.b64encode(output.getvalue()).decode("ascii"),
                mime_type="image/jpeg",
                width=image.width,
                height=image.height,
            )

    @staticmethod
    def _read_ppm(data: bytes) -> tuple[bytes, int, int]:
        index = 0

        def token() -> bytes:
            nonlocal index
            while index < len(data) and data[index] in b" \t\r\n":
                index += 1
            while index < len(data) and data[index] == ord("#"):
                while index < len(data) and data[index] not in b"\r\n":
                    index += 1
                while index < len(data) and data[index] in b" \t\r\n":
                    index += 1
            start = index
            while index < len(data) and data[index] not in b" \t\r\n#":
                index += 1
            if start == index:
                raise ValueError("missing PPM token")
            return data[start:index]

        magic = token()
        if magic not in {b"P3", b"P6"}:
            raise ValueError("unsupported image format")
        width = int(token())
        height = int(token())
        maximum = int(token())
        if not (1 <= width <= 10000 and 1 <= height <= 10000 and 1 <= maximum <= 65535):
            raise ValueError("invalid PPM dimensions")
        if width * height > 20_000_000:
            raise ValueError("image too large")

        if magic == b"P6":
            while index < len(data) and data[index] in b" \t\r\n":
                index += 1
            channels = 3
            sample_width = 2 if maximum > 255 else 1
            expected = width * height * channels * sample_width
            raw = data[index : index + expected]
            if len(raw) != expected:
                raise ValueError("truncated PPM")
            if sample_width == 1:
                pixels = raw
            else:
                pixels = bytes(int.from_bytes(raw[pos : pos + 2], "big") * 255 // maximum for pos in range(0, len(raw), 2))
            return pixels, width, height

        values = [int(token()) for _ in range(width * height * 3)]
        if any(value < 0 or value > maximum for value in values):
            raise ValueError("invalid PPM sample")
        pixels = bytes(value * 255 // maximum for value in values)
        return pixels, width, height

    def _resize_rgb(self, pixels: bytes, width: int, height: int) -> tuple[bytes, int, int]:
        scale = min(1.0, self.max_long_edge / max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if (new_width, new_height) == (width, height):
            return pixels, width, height
        output = bytearray(new_width * new_height * 3)
        for y in range(new_height):
            source_y = min(height - 1, int(y * height / new_height))
            for x in range(new_width):
                source_x = min(width - 1, int(x * width / new_width))
                source = (source_y * width + source_x) * 3
                target = (y * new_width + x) * 3
                output[target : target + 3] = pixels[source : source + 3]
        return bytes(output), new_width, new_height

    @staticmethod
    def _encode_png(pixels: bytes, width: int, height: int) -> bytes:
        rows = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))

        def chunk(kind: bytes, value: bytes) -> bytes:
            return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)

        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
