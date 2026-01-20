import pytest

pytest.importorskip("fiftyone")

import fiftyone.core.media as fomm

from darwin_fiftyone.darwin import _infer_registration_type


class DummyMetadata:
    def __init__(self, mime_type=None):
        self.mime_type = mime_type


class DummySample:
    def __init__(self, media_type, metadata=None):
        self.media_type = media_type
        self.metadata = metadata


def test_infer_registration_type_video_mime() -> None:
    sample = DummySample(
        media_type=fomm.IMAGE,
        metadata=DummyMetadata(mime_type="video/mp4"),
    )

    assert _infer_registration_type(sample) == "video"


def test_infer_registration_type_image_mime() -> None:
    sample = DummySample(
        media_type=fomm.VIDEO,
        metadata=DummyMetadata(mime_type="image/png"),
    )

    assert _infer_registration_type(sample) == "image"


def test_infer_registration_type_missing_mime_falls_back_to_media_type() -> None:
    sample = DummySample(
        media_type=fomm.VIDEO,
        metadata=DummyMetadata(mime_type=None),
    )

    assert _infer_registration_type(sample) == "video"


def test_infer_registration_type_no_metadata_falls_back_to_media_type() -> None:
    sample = DummySample(media_type=fomm.IMAGE, metadata=None)

    assert _infer_registration_type(sample) == "image"
