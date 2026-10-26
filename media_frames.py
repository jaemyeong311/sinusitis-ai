"""Safely decode a small, representative set of image/GIF/MP4 frames.

A sampled video frame is processed by the existing still-image classifier.
This is not a temporal/video model or a validated medical video diagnosis.
"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
import os
import tempfile
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

FRAME_SIZE = (96, 96)
MAX_IMAGE_PIXELS = 16_000_000
MAX_GIF_FRAMES = 2_000
MAX_VIDEO_FRAMES = 9_000
MAX_VIDEO_SECONDS = 300
SAMPLE_FRAMES = 5

CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/gif": "GIF",
    "video/mp4": "MP4",
}


class MediaError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SampledFrame:
    index: int
    image: Image.Image
    time_seconds: float | None = None


def _sample_indices(total: int) -> list[int]:
    if total < 1:
        raise MediaError(400, "영상에서 프레임을 찾지 못했습니다.")
    count = min(total, SAMPLE_FRAMES)
    if count == 1:
        return [0]
    return sorted({
        round(i * (total - 1) / (count - 1))
        for i in range(count)
    })


def _resize(frame: Image.Image) -> Image.Image:
    return frame.convert("RGB").resize(FRAME_SIZE, Image.Resampling.BILINEAR)


def _decode_pillow(data: bytes, content_type: str):
    expected_format = CONTENT_TYPES[content_type]
    signature_ok = {
        "image/jpeg": data.startswith(b"\xff\xd8"),
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/gif": data[:6] in (b"GIF87a", b"GIF89a"),
    }[content_type]
    if not signature_ok:
        raise MediaError(415, "파일 확장자와 실제 이미지 형식이 일치하지 않습니다.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format != expected_format:
                    raise MediaError(415, "파일의 실제 이미지 형식이 올바르지 않습니다.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise MediaError(413, "이미지는 1,600만 픽셀 이하여야 합니다.")
                total = getattr(source, "n_frames", 1)
                if total > MAX_GIF_FRAMES:
                    raise MediaError(413, "GIF 프레임은 2,000개 이하여야 합니다.")

                frames = []
                for index in _sample_indices(total):
                    source.seek(index)
                    frames.append(SampledFrame(index, _resize(source)))
                return frames, total

    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise MediaError(413, "이미지 해상도가 너무 큽니다.")
    except (UnidentifiedImageError, OSError, SyntaxError, EOFError, ValueError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError(400, "이미지를 읽을 수 없거나 파일이 손상됐습니다.") from exc


def _decode_mp4(data: bytes):
    if len(data) < 12 or data[4:8] != b"ftyp":
        raise MediaError(415, "실제 MP4 파일이 아닙니다.")

    import cv2  # Imported only for MP4 processing.

    path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            temp.write(data)
            path = temp.name

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise MediaError(400, "MP4 영상을 열 수 없습니다.")

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if math.isfinite(fps) and fps > 0 else None

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0 and width * height > MAX_IMAGE_PIXELS:
            raise MediaError(413, "영상 프레임 해상도가 너무 큽니다.")
        if total > MAX_VIDEO_FRAMES or (
            total > 0 and fps and total / fps > MAX_VIDEO_SECONDS
        ):
            raise MediaError(413, "MP4는 최대 5분 / 9,000프레임까지 지원합니다.")

        frames = []
        # Some MP4 containers do not publish a frame count. In that case
        # inspect only the first five frames, rather than decoding unboundedly.
        indices = _sample_indices(total) if total > 0 else list(range(SAMPLE_FRAMES))
        for index in indices:
            if total > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, bgr = capture.read()
            if not success or bgr is None:
                if total > 0:
                    continue
                break
            if bgr.shape[0] * bgr.shape[1] > MAX_IMAGE_PIXELS:
                raise MediaError(413, "영상 프레임 해상도가 너무 큽니다.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = _resize(Image.fromarray(rgb))
            seconds = round(index / fps, 3) if fps else None
            frames.append(SampledFrame(index, image, seconds))

        if not frames:
            raise MediaError(400, "MP4에서 분석 가능한 프레임을 읽지 못했습니다.")
        return frames, total if total > 0 else None

    except (cv2.error, OSError) as exc:
        raise MediaError(400, "MP4 파일을 처리하지 못했습니다.") from exc
    finally:
        if capture is not None:
            capture.release()
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass


def extract_frames(data: bytes, content_type: str):
    """Return (sampled frames, original frame count or None)."""
    if content_type not in CONTENT_TYPES:
        raise MediaError(415, "JPG, PNG, GIF, MP4 파일만 업로드해주세요.")
    if content_type == "video/mp4":
        return _decode_mp4(data)
    return _decode_pillow(data, content_type)
