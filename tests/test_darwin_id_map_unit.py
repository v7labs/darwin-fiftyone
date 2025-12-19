import importlib
import json
import os
from pathlib import Path
import sys
import types


def _install_stub_modules(*, import_calls):
    """Installs stub `darwin` and `fiftyone` modules into `sys.modules`.

    This allows us to import `darwin_fiftyone.darwin` without requiring the real
    Darwin SDK / FiftyOne (which are heavy and not needed for these unit tests).
    """

    # ---- darwin stubs ----
    darwin_mod = types.ModuleType("darwin")

    class _NotFound(Exception):
        pass

    class _ValidationError(Exception):
        pass

    darwin_mod.exceptions = types.SimpleNamespace(
        NotFound=_NotFound,
        ValidationError=_ValidationError,
    )

    darwin_importer_mod = types.ModuleType("darwin.importer")

    def _import_annotations(dataset, parser, files, append=False, class_prompt=False):
        import_calls.append(
            {
                "dataset": dataset,
                "parser": parser,
                "files": list(files),
                "append": append,
                "class_prompt": class_prompt,
            }
        )

    darwin_importer_mod.import_annotations = _import_annotations

    darwin_client_mod = types.ModuleType("darwin.client")

    class _Client:
        @classmethod
        def from_api_key(cls, api_key):
            return cls()

    darwin_client_mod.Client = _Client

    def _get_importer(name):
        return f"parser:{name}"

    # `darwin_fiftyone.darwin` imports get_importer from darwin.importer
    darwin_importer_mod.get_importer = _get_importer

    sys.modules["darwin"] = darwin_mod
    sys.modules["darwin.importer"] = darwin_importer_mod
    sys.modules["darwin.client"] = darwin_client_mod

    # ---- requests stub ----
    # Importing real `requests` can fail in some restricted environments due to
    # SSL CA bundle loading. The darwin backend imports requests at module load,
    # but our unit tests don't exercise HTTP calls, so a stub is sufficient.
    requests_mod = types.ModuleType("requests")

    class _Response:
        ok = True

        def __init__(self, text="", status_code=200):
            self.text = text
            self.status_code = status_code

        def json(self):
            return {}

        def raise_for_status(self):
            return None

    def _noop(*args, **kwargs):
        return _Response()

    requests_mod.Response = _Response
    requests_mod.get = _noop
    requests_mod.post = _noop
    requests_mod.put = _noop
    requests_mod.patch = _noop
    requests_mod.delete = _noop
    sys.modules["requests"] = requests_mod

    # ---- fiftyone stubs ----
    # Create package modules so `import fiftyone.core.labels as fol` works
    fiftyone_pkg = types.ModuleType("fiftyone")
    fiftyone_core_pkg = types.ModuleType("fiftyone.core")
    fiftyone_utils_pkg = types.ModuleType("fiftyone.utils")

    fiftyone_utils_annotations_mod = types.ModuleType("fiftyone.utils.annotations")

    class _AnnotationBackendConfig:
        def __init__(self, name=None, label_schema=None, media_field="filepath", **kwargs):
            self.name = name
            self.label_schema = label_schema
            self.media_field = media_field

    class _AnnotationBackend:
        def __init__(self, config=None):
            self.config = config

        def connect_to_api(self):
            return self._connect_to_api()

    class _AnnotationAPI:
        def __init__(self):
            pass

    class _AnnotationResults:
        def __init__(self, samples, config, anno_key, id_map, backend=None):
            self.samples = samples
            self.config = config
            self.anno_key = anno_key
            self.id_map = id_map
            self.backend = backend

    fiftyone_utils_annotations_mod.AnnotationBackendConfig = _AnnotationBackendConfig
    fiftyone_utils_annotations_mod.AnnotationBackend = _AnnotationBackend
    fiftyone_utils_annotations_mod.AnnotationAPI = _AnnotationAPI
    fiftyone_utils_annotations_mod.AnnotationResults = _AnnotationResults

    fiftyone_core_media_mod = types.ModuleType("fiftyone.core.media")
    fiftyone_core_media_mod.IMAGE = "image"
    fiftyone_core_media_mod.VIDEO = "video"

    fiftyone_core_metadata_mod = types.ModuleType("fiftyone.core.metadata")

    class _ImageMetadata:
        def __init__(self, width=10, height=10):
            self.width = width
            self.height = height

        @classmethod
        def build_for(cls, filepath):
            return cls()

    class _VideoMetadata:
        def __init__(self, frame_width=10, frame_height=10):
            self.frame_width = frame_width
            self.frame_height = frame_height

        @classmethod
        def build_for(cls, filepath):
            return cls()

    fiftyone_core_metadata_mod.ImageMetadata = _ImageMetadata
    fiftyone_core_metadata_mod.VideoMetadata = _VideoMetadata

    fiftyone_core_labels_mod = types.ModuleType("fiftyone.core.labels")
    # Only needed for type references; not used by these unit tests
    fiftyone_core_labels_mod.Keypoint = type("Keypoint", (), {})
    fiftyone_core_labels_mod.Detection = type("Detection", (), {})
    fiftyone_core_labels_mod.Classification = type("Classification", (), {})
    fiftyone_core_labels_mod.Polyline = type("Polyline", (), {})

    sys.modules["fiftyone"] = fiftyone_pkg
    sys.modules["fiftyone.core"] = fiftyone_core_pkg
    sys.modules["fiftyone.utils"] = fiftyone_utils_pkg
    sys.modules["fiftyone.utils.annotations"] = fiftyone_utils_annotations_mod
    sys.modules["fiftyone.core.media"] = fiftyone_core_media_mod
    sys.modules["fiftyone.core.metadata"] = fiftyone_core_metadata_mod
    sys.modules["fiftyone.core.labels"] = fiftyone_core_labels_mod


class _FakeAnno:
    def __init__(self, _id, label="x", confidence=None, index=None, attributes=None):
        self.id = _id
        self.label = label
        self.confidence = confidence
        self.index = index
        self.attributes = attributes or {}

    def __getitem__(self, key):
        return self.attributes[key]


class _FakeLabelContainer:
    """Mimics FO containers that support `container[label_type] -> list[Label]`."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __getitem__(self, key):
        return self._mapping[key]


class _FakeFrame:
    def __init__(self, _id, data):
        self.id = _id
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class _FakeSample:
    def __init__(self, _id, media_type, filepath="/tmp/a.jpg", data=None, frames=None):
        self.id = _id
        self.media_type = media_type
        self.metadata = None
        self.frames = frames or {}
        self._data = {"filepath": filepath}
        if data:
            self._data.update(data)

    def __getitem__(self, key):
        return self._data[key]


def _import_darwin_module(import_calls):
    # Ensure clean import each time
    for m in [
        "darwin_fiftyone.darwin",
    ]:
        sys.modules.pop(m, None)

    # Avoid darwin_fiftyone/darwin.py trying to write logs outside the workspace
    os.environ["DISABLE_DARWIN_FIFTYONE_LOGGING"] = "true"

    _install_stub_modules(import_calls=import_calls)
    return importlib.import_module("darwin_fiftyone.darwin")


def test_id_map_images_is_per_field():
    import_calls = []
    mod = _import_darwin_module(import_calls)

    api = mod.DarwinAPI.__new__(mod.DarwinAPI)
    api._convert_image_annotation_to_v7 = lambda *args, **kwargs: []

    backend = types.SimpleNamespace(config=types.SimpleNamespace(atts=None, Groups=False))

    sample_id = "s1"
    sample = _FakeSample(
        _id=sample_id,
        media_type=mod.fomm.IMAGE,
        data={
            "cls": _FakeAnno("a1"),
            "clses": _FakeLabelContainer({"classifications": [_FakeAnno("b1"), _FakeAnno("b2")]}),
            "dets": _FakeLabelContainer({"detections": [_FakeAnno("c1"), _FakeAnno("c2")]}),
        },
    )

    label_schema = {
        "cls": {"type": "classification", "classes": ["a"]},
        "clses": {"type": "classifications", "classes": ["a"]},
        "dets": {"type": "detections", "classes": ["a"]},
    }

    id_map = {}
    api._convert_image_annotations_to_v7(
        sample,
        frame_size=(10, 10),
        label_schema=label_schema,
        id_map=id_map,
        backend=backend,
        slot_name="0",
        frame_val=None,
    )

    assert id_map == {
        "cls": {sample_id: "a1"},
        "clses": {sample_id: ["b1", "b2"]},
        "dets": {sample_id: ["c1", "c2"]},
    }


def test_id_map_videos_is_per_field_per_frame():
    import_calls = []
    mod = _import_darwin_module(import_calls)

    api = mod.DarwinAPI.__new__(mod.DarwinAPI)
    api._convert_image_annotation_to_v7 = lambda *args, **kwargs: []

    backend = types.SimpleNamespace(config=types.SimpleNamespace(atts=None, Groups=False))

    sample_id = "sv"
    frame1_id = "f1"
    frame2_id = "f2"
    sample = _FakeSample(
        _id=sample_id,
        media_type=mod.fomm.VIDEO,
        filepath="/tmp/a.mp4",
        frames={
            1: _FakeFrame(
                frame1_id,
                {
                    "dets": _FakeLabelContainer({"detections": [_FakeAnno("d1"), _FakeAnno("d2")]}),
                    "cls": _FakeAnno("c1"),
                },
            ),
            2: _FakeFrame(
                frame2_id,
                {
                    "dets": _FakeLabelContainer({"detections": [_FakeAnno("d3")]}),
                    "cls": _FakeAnno("c2"),
                },
            ),
        },
    )

    label_schema = {
        "frames.dets": {"type": "detections", "classes": ["a"]},
        "frames.cls": {"type": "classification", "classes": ["a"]},
    }

    id_map = {}
    # Simulate the per-frame calls made by _upload_annotations()
    api._convert_image_annotations_to_v7(
        sample,
        frame_size=(10, 10),
        label_schema=label_schema,
        id_map=id_map,
        backend=backend,
        slot_name="0",
        frame_val=sample.frames[1],
    )
    api._convert_image_annotations_to_v7(
        sample,
        frame_size=(10, 10),
        label_schema=label_schema,
        id_map=id_map,
        backend=backend,
        slot_name="0",
        frame_val=sample.frames[2],
    )

    assert id_map == {
        "frames.dets": {sample_id: {frame1_id: ["d1", "d2"], frame2_id: ["d3"]}},
        "frames.cls": {sample_id: {frame1_id: "c1", frame2_id: "c2"}},
    }


def test_upload_annotations_imports_once():
    import_calls = []
    mod = _import_darwin_module(import_calls)

    api = mod.DarwinAPI.__new__(mod.DarwinAPI)

    # Force deterministic conversion output (and avoid using real label objects)
    api._convert_image_annotations_to_v7 = lambda *args, **kwargs: [
        {"name": "x", "slot_names": ["0"], "tag": {}}
    ]

    dataset = object()
    backend = types.SimpleNamespace(config=types.SimpleNamespace(Groups=False))

    samples = [
        _FakeSample(_id="s1", media_type=mod.fomm.IMAGE, filepath="/tmp/a.jpg"),
        _FakeSample(_id="s2", media_type=mod.fomm.IMAGE, filepath="/tmp/b.jpg"),
    ]

    label_schema = {
        "cls": {"type": "classification", "classes": ["a"]},
    }

    id_map, frame_id_map = api._upload_annotations(
        label_schema=label_schema,
        samples=samples,
        media_field="filepath",
        dataset=dataset,
        backend=backend,
    )

    assert isinstance(id_map, dict)
    assert frame_id_map == {}
    assert len(import_calls) == 1, "Expected a single import_annotations() call per run"
    assert len(import_calls[0]["files"]) == len(samples), "Expected one JSON file per sample"


def test_upload_annotations_no_duplicate_import_when_multiple_fields():
    """Regression: users reported duplicate annotations in FiftyOne.

    One cause was importing the same per-sample JSON payload multiple times in a
    single run (e.g. once per label field). This test asserts we only import
    once, even when multiple label fields are present.
    """
    import_calls = []
    mod = _import_darwin_module(import_calls)

    api = mod.DarwinAPI.__new__(mod.DarwinAPI)

    # Make the conversion deterministic + include the label ID so we can detect duplicates
    def _to_v7(self, annotation, label_type, frame_size, sample, backend, slot_name="0", attributes=None):
        return [{"name": f"id:{annotation.id}", "slot_names": [slot_name], "tag": {}}]

    api._convert_image_annotation_to_v7 = types.MethodType(_to_v7, api)

    backend = types.SimpleNamespace(config=types.SimpleNamespace(atts=None, Groups=False, item_name_annotation=False))

    sample_id = "s1"
    sample = _FakeSample(
        _id=sample_id,
        media_type=mod.fomm.IMAGE,
        filepath="/tmp/a.jpg",
        data={
            "cls": _FakeAnno("a1"),
            "clses": _FakeLabelContainer({"classifications": [_FakeAnno("b1"), _FakeAnno("b2")]}),
            "dets": _FakeLabelContainer({"detections": [_FakeAnno("c1"), _FakeAnno("c2")]}),
        },
    )

    label_schema = {
        "cls": {"type": "classification", "classes": ["a"]},
        "clses": {"type": "classifications", "classes": ["a"]},
        "dets": {"type": "detections", "classes": ["a"]},
    }

    id_map, _ = api._upload_annotations(
        label_schema=label_schema,
        samples=[sample],
        media_field="filepath",
        dataset=object(),
        backend=backend,
    )

    # Critical assertion: only one import per run
    assert len(import_calls) == 1, "Expected exactly one import_annotations() call per run"

    # Also assert the generated JSON has no duplicated annotations
    assert len(import_calls[0]["files"]) == 1
    json_path = import_calls[0]["files"][0]
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    names = [a["name"] for a in payload["annotations"]]
    assert len(names) == 5  # 1 classification + 2 classifications + 2 detections
    assert len(set(names)) == len(names), "Expected no duplicated annotations in exported JSON"

    # Sanity check: id_map is per field (not mixed)
    assert id_map["cls"][sample_id] == "a1"
    assert id_map["clses"][sample_id] == ["b1", "b2"]
    assert id_map["dets"][sample_id] == ["c1", "c2"]


def test_upload_annotations_video_frame_id_map_preserves_id_types():
    """Ensure `frame_id_map` uses the same sample_id type as `item_sample_map` emits.

    In the real integration, `item_sample_map` stores raw `sample.id` values, and
    `download_annotations()` later indexes `frame_id_map[sample_id]`. So we must
    not stringify `sample.id` or `frame.id` in `frame_id_map`.
    """
    import_calls = []
    mod = _import_darwin_module(import_calls)

    api = mod.DarwinAPI.__new__(mod.DarwinAPI)

    # We don't need to generate actual annotations for this test
    api._convert_image_annotations_to_v7 = lambda *args, **kwargs: []

    backend = types.SimpleNamespace(config=types.SimpleNamespace(Groups=False))

    class SampleId:
        def __init__(self, v):
            self.v = v

        def __repr__(self):
            return f"SampleId({self.v})"

        def __str__(self):
            return f"SID:{self.v}"

    class FrameId:
        def __init__(self, v):
            self.v = v

        def __repr__(self):
            return f"FrameId({self.v})"

        def __str__(self):
            return f"FID:{self.v}"

    sample_id_obj = SampleId("abc")
    frame1_id_obj = FrameId("f1")

    sample = _FakeSample(
        _id=sample_id_obj,
        media_type=mod.fomm.VIDEO,
        filepath="/tmp/a.mp4",
        frames={
            1: _FakeFrame(frame1_id_obj, {"dets": _FakeLabelContainer({"detections": []}), "cls": None}),
        },
    )

    label_schema = {"frames.cls": {"type": "classification", "classes": ["a"]}}

    _, frame_id_map = api._upload_annotations(
        label_schema=label_schema,
        samples=[sample],
        media_field="filepath",
        dataset=object(),
        backend=backend,
    )

    assert sample_id_obj in frame_id_map
    assert str(sample_id_obj) not in frame_id_map
    assert frame_id_map[sample_id_obj]["1"] is frame1_id_obj
    assert str(frame_id_map[sample_id_obj]["1"]) == str(frame1_id_obj)


