import pytest

pytest.importorskip("fiftyone")

import fiftyone.core.media as fomm

from darwin_fiftyone.darwin import _get_readonly_registration_slot


class DummyMetadata:
    def __init__(self, mime_type=None):
        self.mime_type = mime_type


class DummySample:
    def __init__(self, darwin_metadata, media_type, metadata=None):
        self.darwin_metadata = darwin_metadata
        self.media_type = media_type
        self.metadata = metadata

    def has_field(self, field_name):
        return field_name == "darwin_metadata"


def test_get_readonly_registration_slot_overrides_type_from_mime() -> None:
    sample = DummySample(
        darwin_metadata={
            "readonly_registration_payload": {
                "type": "video",
                "name": "image.jpg",
                "path": "/",
                "storage_key": "bucket/image.jpg",
                "total_size_bytes": 123,
            }
        },
        media_type=fomm.IMAGE,
        metadata=DummyMetadata(mime_type="image/jpeg"),
    )

    payload = _get_readonly_registration_slot(sample)

    assert payload["type"] == "image"
    assert payload["size_bytes"] == 123
    assert "name" not in payload
    assert "path" not in payload
    assert "storage_key" not in payload


def test_get_readonly_registration_slot_defaults_type_from_media_type() -> None:
    sample = DummySample(
        darwin_metadata={
            "readonly_registration_payload": {
                "name": "video.mp4",
                "path": "/",
                "storage_key": "bucket/video.mp4",
            }
        },
        media_type=fomm.VIDEO,
        metadata=None,
    )

    payload = _get_readonly_registration_slot(sample)

    assert payload["type"] == "video"
