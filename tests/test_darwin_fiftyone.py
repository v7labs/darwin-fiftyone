"""
Tests for the :mod:`darwin_fiftyone` module.
Runs against the IRL Darwin production environment.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
=======
"""
from dotenv import load_dotenv

# Load environment variables from .env file at the start
load_dotenv(override=True)

import os
import logging

import pytest
import requests

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.labels as foul
from fiftyone import ViewField as F

import darwin_fiftyone
import darwin.future.core.workflows as dfcw
import darwin.future.core.client as dfcc
from darwin.cli_functions import remove_remote_dataset
from darwin.client import Client

logger = logging.getLogger(__name__)

team_slug = os.environ["FIFTYONE_DARWIN_TEAM_SLUG"]
api_key = os.environ["FIFTYONE_DARWIN_API_KEY"]
test_bucket = os.environ["FIFTYONE_DARWIN_TEST_BUCKET"]
test_external = os.environ["FIFTYONE_DARWIN_TEST_EXTERNAL_STORAGE"]


@pytest.fixture(scope="session", autouse=True)
def set_darwin_base_url():
    os.environ["DARWIN_BASE_URL"] = "https://darwin.irl.v7labs.com"


@pytest.fixture()
def backend_config():
    fo.annotation_config.backends["darwin"]["api_key"] = api_key


@pytest.fixture()
def setup_quickstart_empty():
    """dataset with no labels"""

    dsn = "fo-v7-test-qs2-empty"
    if fo.dataset_exists(dsn):
        fo.delete_dataset(dsn)

    dataset = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name=dsn)
    dataset.delete_sample_fields(["ground_truth", "predictions"])
    dataset.persistent = True
    return dataset


@pytest.fixture()
def setup_quickstart():
    """dataset with 2 samps, classification, classifications, detections"""

    dsn = "fo-v7-test-qs2"
    if fo.dataset_exists(dsn):
        fo.delete_dataset(dsn)

    dataset = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name=dsn)

    s0 = dataset.first()
    s1 = dataset.last()
    s0["cls"] = fo.Classification(label="cls0")
    s1["cls"] = fo.Classification(label="cls1")
    s0["clses"] = fo.Classifications(
        classifications=[
            fo.Classification(label="clses0"),
            fo.Classification(label="clses1"),
        ]
    )
    s1["clses"] = fo.Classifications(
        classifications=[fo.Classification(label="clses2")]
    )
    s0.save()
    s1.save()
    dataset.persistent = True

    return dataset, "cls", "clses", "ground_truth"


@pytest.fixture()
def setup_quickstart_external():
    """dataset with 3 samps, detections, external bucket"""

    dsn = "fo-v7-test-qs3-external"
    if fo.dataset_exists(dsn):
        fo.delete_dataset(dsn)

    dataset = foz.load_zoo_dataset("quickstart", max_samples=3, dataset_name=dsn)
    dataset.compute_metadata()

    fps = dataset.values("filepath")
    fps = [os.path.join(test_bucket, os.path.basename(f)) for f in fps]
    dataset.set_values("filepath", fps)
    dataset.persistent = True

    return dataset, "ground_truth"


@pytest.fixture()
def setup_quickstart_external_video():
    """dataset with 2 video samps, detections, external bucket"""

    dsn = "fo-v7-test-qsv2-external"
    if fo.dataset_exists(dsn):
        fo.delete_dataset(dsn)

    dataset = foz.load_zoo_dataset("quickstart-video", max_samples=2, dataset_name=dsn)
    dataset.compute_metadata()
    dataset.ensure_frames()

    fps = dataset.values("filepath")
    fps = [os.path.join(test_bucket, os.path.basename(f)) for f in fps]
    dataset.set_values("filepath", fps)

    dataset.persistent = True

    return dataset, "frames.detections"


@pytest.fixture()
def setup_coco():
    """dataset with 2 samps, instance segs, polylines, keypts"""

    dsn = "fo-v7-test-coco2"
    if fo.dataset_exists(dsn):
        fo.delete_dataset(dsn)

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        max_samples=2,
        split="validation",
        label_types=["segmentations"],
        dataset_name=dsn,
    )

    foul.instances_to_polylines(dataset, "ground_truth", "polylines")
    foul.instances_to_polylines(
        dataset, "ground_truth", "polylines_unfilled", filled=False
    )

    # add keypoints
    for sample in dataset.iter_samples(progress=True, autosave=True):
        plylines = sample.polylines.polylines
        keypts = [fo.Keypoint(points=x.points[0], label=x.label) for x in plylines]
        keypts = fo.Keypoints(keypoints=keypts)
        sample["keypoints"] = keypts

    dataset.persistent = True

    return dataset, "ground_truth", "polylines", "polylines_unfilled", "keypoints"


def get_dataset_id_from_slug(dataset_slug):
    url = "https://darwin.irl.v7labs.com/api/datasets"
    headers = {"accept": "application/json", "Authorization": f"ApiKey {api_key}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    datasets = response.json()
    for dataset in datasets:
        if dataset["slug"] == dataset_slug:
            return dataset["id"]
    raise ValueError(f"Dataset with slug {dataset_slug} not found")


def test_annotate_existing_class_det(setup_quickstart):
    """Test edit {existing field} x {classification, classifications, detections}

    This creates/loads a dataset with existing labels and launches 3 annotation jobs

    Annotate by hand and then load results with test_load_existing_class_det
    """
    dataset, fld_class, fld_classes, fld_dets = setup_quickstart
    anno_key = "class_exist"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_class
    dataset.annotate(
        anno_key,
        label_field=label_field,
        attributes=["iscrowd"],
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "classes_exist"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_classes
    dataset.annotate(
        anno_key,
        label_field=label_field,
        attributes=["iscrowd"],
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "dets_exist"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_dets
    dataset.annotate(
        anno_key,
        label_field=label_field,
        attributes=["iscrowd"],
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


def test_load_existing_class_det():
    """See test_annotate_existing_class_det"""

    dataset_name = "fo-v7-test-qs2"
    dataset = fo.load_dataset(dataset_name)
    for anno_key in ["class_exist", "classes_exist", "dets_exist"]:
        dataset.load_annotations(anno_key)


def test_annotate_existing_polyline_keypts(setup_coco):
    """Test edit {existing field} x {polyline, polyline_unfilled, keypoints}

    This creates/loads a dataset with existing labels and launches annotation jobs

    Annotate by hand and then load results with test_load_existing_polyline_keypts
    """

    dataset, fld_seg, fld_poly, fld_poly_unfilled, fld_keypts = setup_coco

    anno_key = "poly_exist"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_poly
    dataset.annotate(
        anno_key,
        label_field=label_field,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "poly_unfilled_exist"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_poly_unfilled
    dataset.annotate(
        anno_key,
        label_field=label_field,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "keypts_exist"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_keypts
    dataset.annotate(
        anno_key,
        label_field=label_field,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


def test_load_existing_polyline_keypts():
    """See test_annotate_existing_polyline_keypts"""

    dataset_name = "fo-v7-test-coco2"
    dataset = fo.load_dataset(dataset_name)
    for anno_key in ["poly_exist", "keypts_exist"]:

        dataset.load_annotations(anno_key)


def test_annotate_new(setup_quickstart_empty):
    """Test annotate {new field} x
        {classification, classifications, detections, polylines, keypoints}

    This creates/loads a dataset with no labels and launches annotation jobs

    Annotate by hand and then load results with test_load_new
    """

    dataset = setup_quickstart_empty

    anno_key = "class_new"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = anno_key
    classes = ["class0", "class1"]
    dataset.annotate(
        anno_key,
        label_field=label_field,
        label_type="classification",
        classes=classes,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "classes_new"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = anno_key
    classes = ["classes0", "classes1", "classes2"]
    dataset.annotate(
        anno_key,
        label_field=label_field,
        label_type="classifications",
        classes=classes,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "dets_new"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = anno_key
    classes = ["dets0", "dets1"]
    dataset.annotate(
        anno_key,
        label_field=label_field,
        label_type="detections",
        classes=classes,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "polygons_new"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = anno_key
    classes = ["poly0", "poly1", "poly2"]
    dataset.annotate(
        anno_key,
        label_field=label_field,
        label_type="polygons",
        classes=classes,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "kpts_new"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = anno_key
    classes = ["kpts0", "kpts1", "kpts2"]
    dataset.annotate(
        anno_key,
        label_field=label_field,
        label_type="keypoints",
        classes=classes,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


def test_load_new():
    """See test_load_new"""

    dataset_name = "fo-v7-test-qs2-empty"
    anno_keys = ["class_new", "classes_new", "dets_new", "polygons_new", "kpts_new"]

    dataset = fo.load_dataset(dataset_name)
    for anno_key in anno_keys:

        dataset.load_annotations(anno_key)


def test_annotate_schema(setup_quickstart):
    """
    side effect, tests new classes for existing dets
    """

    dataset, fld_class, fld_classes, fld_dets = setup_quickstart

    # from doc example
    label_schema0 = {
        "new_classifications": {
            "type": "classifications",
            "classes": ["dog", "cat", "person"],
        }
    }

    label_schema1 = {
        fld_dets: {
            "type": "detections",
            "classes": ["detclass0", "detclass1", "detclass2"],
        }
    }

    anno_key = "class_new_schema"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    dataset.annotate(
        anno_key,
        label_schema=label_schema0,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )

    anno_key = "class_existing_schema"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    dataset.annotate(
        anno_key,
        label_schema=label_schema1,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


def test_load_schema():
    """See"""

    dataset_name = "fo-v7-test-qs2"

    dataset = fo.load_dataset(dataset_name)
    anno_keys = dataset.list_annotation_runs()
    for anno_key in anno_keys:

        dataset.load_annotations(anno_key)


def test_annotate_external_media(setup_quickstart_external):
    """Test edit existing detection field with external media

    Annotate by hand and then load results with test_load_external_media
    """

    dataset, fld_gt = setup_quickstart_external
    anno_key = "externalmedia"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_gt
    dataset.annotate(
        anno_key,
        label_field=label_field,
        attributes=["iscrowd"],
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        external_storage=test_external,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


def test_annotate_external_media_video(setup_quickstart_external_video):
    """Test edit existing detection field with external media (video dataset)

    Annotate by hand and then load results with test_load_external_media_video
    """

    dataset, fld_gt = setup_quickstart_external_video
    anno_key = "externalmedia"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = fld_gt
    dataset.annotate(
        anno_key,
        label_field=label_field,
        attributes=["iscrowd"],
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        external_storage=test_external,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


def test_load_external_media():
    """See test_annotate_external_media"""

    dataset_name = "fo-v7-test-qs3-external"
    dataset = fo.load_dataset(dataset_name)
    for anno_key in ["externalmedia"]:
        dataset.load_annotations(anno_key)


def test_load_external_media_video():
    """See test_annotate_external_media_video"""

    dataset_name = "fo-v7-test-qsv2-external"
    dataset = fo.load_dataset(dataset_name)
    for anno_key in ["externalmedia"]:
        dataset.load_annotations(anno_key)


@pytest.mark.usefixtures("mocker")
def test_cleanup(mocker, setup_quickstart_empty):
    """Test backend cleanup

    Launch ann job then call load with cleanup kwarg
    """

    # Mock user input for cleanup confirmation
    mocker.patch("builtins.input", return_value="y")
    dataset = setup_quickstart_empty
    anno_key = "class_cleanup"
    v7_dataset_slug = f"{dataset.name}-{anno_key}"
    label_field = anno_key
    classes = ["class0", "class1"]
    dataset.annotate(
        anno_key,
        label_field=label_field,
        label_type="classification",
        classes=classes,
        backend="darwin",
        dataset_slug=v7_dataset_slug,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )
    dataset.load_annotations(anno_key, cleanup=True)


def test_annotate_doc_example():
    dataset_name = "fo-v7-test-annotation-example"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = foz.load_zoo_dataset("quickstart", dataset_name=dataset_name)
    dataset.persistent = True
    dataset.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")

    # Step 2: Locate a subset of your data requiring annotation

    # Create a view that contains only high confidence false positive model
    # predictions, with samples containing the most false positives first
    most_fp_view = dataset.filter_labels(
        "predictions", (F("confidence") > 0.8) & (F("eval") == "fp")
    ).sort_by(F("predictions.detections").length(), reverse=True)

    # Retrieve the sample with the most high confidence false positives
    sample_id = most_fp_view.first().id
    view = dataset.select(sample_id)

    # Step 3: Send samples to V7

    # A unique identifier for this run
    anno_key = "v7_basic_recipe"

    label_schema = {
        "new_ground_truth": {
            "type": "detections",
            "classes": dataset.distinct("ground_truth.detections.label"),
        },
    }

    view.annotate(
        anno_key,
        backend="darwin",
        label_schema=label_schema,
        launch_editor=True,
        dataset_slug=dataset_name,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


@pytest.mark.usefixtures("mocker")
def test_load_doc_example(mocker):
    anno_key = "v7_basic_recipe"

    # Step 5: Merge annotations back into FiftyOne dataset
    dataset = fo.load_dataset("fo-v7-test-annotation-example")
    dataset.load_annotations(anno_key)

    # Step 6: Cleanup
    # Mock user input for cleanup confirmation
    mocker.patch("builtins.input", return_value="y")

    # Delete tasks from V7
    results = dataset.load_annotation_results(anno_key)
    results.cleanup()

    # Delete run record (not the labels) from FiftyOne
    dataset.delete_annotation_run(anno_key)


def test_annotate_vid_example():
    """Tests basic video annotation"""
    dataset = foz.load_zoo_dataset(
        "quickstart-video", dataset_name="darwin-annotation-example-70"
    )

    anno_key = "video_test_1"

    dataset.annotate(
        anno_key,
        label_field="frames.detections",
        atts=["iscrowd", "test1", "test2"],
        launch_editor=True,
        backend="darwin",
        dataset_slug="quickstart-example-video-test",
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )
    dataset.load_annotations(anno_key)


def test_multi_slot():
    """Tests multi-slot annotation"""
    group_dataset = foz.load_zoo_dataset("quickstart-groups")
    group_ids = group_dataset.take(3).values("group.id")
    group_view = group_dataset.select_groups(group_ids)
    groups = group_view.select_group_slices(media_type="image")

    anno_key = "multislot_test_1"

    label_schema = {
        "ground_truth": {
            "type": "detections",
            "classes": ["Car"],
        }
    }

    groups.annotate(
        anno_key,
        label_schema=label_schema,
        launch_editor=True,
        backend="darwin",
        dataset_slug="group-test",
        Groups=True,
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )
    group_dataset.load_annotations(anno_key)


def test_annotate_full_label_schema(setup_quickstart):
    """Tests all avialable class & property permutations"""

    dsn = "fo-v7-test-full-label-schema"
    if fo.dataset_exists(dsn):
        fo.delete_dataset(dsn)

    dataset = foz.load_zoo_dataset("quickstart", max_samples=2, dataset_name=dsn)
    dataset.delete_sample_fields(["predictions"])

    label_schema = {
        "ground_truth": {
            "type": "detections",
            "classes": [
                {
                    "classes": ["class1", "class2"],
                    "attributes": {
                        "single_select_section_level": {
                            "type": "single_select",
                            "granularity": "section",
                            "values": ["val1", "val2"],
                        },
                        "single_select_section_level_required": {
                            "type": "single_select",
                            "granularity": "section",
                            "required": True,
                            "values": ["val1", "val2"],
                        },
                        "multi_select_section_level": {
                            "type": "multi_select",
                            "granularity": "section",
                            "values": ["val1", "val2"],
                        },
                        "multi_select_section_level_required": {
                            "type": "multi_select",
                            "granularity": "section",
                            "required": True,
                            "values": ["val1", "val2"],
                        },
                        "single_select_annotation_level": {
                            "type": "single_select",
                            "granularity": "annotation",
                            "values": ["val1", "val2"],
                        },
                        "single_select_annotation_level_required": {
                            "type": "single_select",
                            "granularity": "annotation",
                            "required": True,
                            "values": ["val1", "val2"],
                        },
                        "multi_select_annotation_level": {
                            "type": "multi_select",
                            "granularity": "annotation",
                            "values": ["val1", "val2"],
                        },
                        "multi_select_annotation_level_required": {
                            "type": "multi_select",
                            "granularity": "annotation",
                            "required": True,
                            "values": ["val1", "val2"],
                        },
                        "my_ids": {"type": "instance_id"},
                        "my_text": {"type": "text"},
                    },
                },
                "class3",
                "class4",
            ],
            "attributes": {
                "single_select_item_level": {
                    "type": "single_select",
                    "granularity": "item",
                    "values": ["val1", "val2"],
                },
                "single_select_item_level_required": {
                    "type": "single_select",
                    "granularity": "item",
                    "required": True,
                    "values": ["val1", "val2"],
                },
                "multi_select_item_level": {
                    "type": "multi_select",
                    "granularity": "item",
                    "values": ["val1", "val2"],
                },
                "multi_select_item_level_required": {
                    "type": "multi_select",
                    "granularity": "item",
                    "required": True,
                    "values": ["val1", "val2"],
                },
                "text_item_prop": {"type": "text", "granularity": "item"},
                "text_item_prop_required": {
                    "type": "text",
                    "granularity": "item",
                    "required": True,
                },
            },
        },
    }

    dataset.annotate(
        "full_label_schema",
        label_schema=label_schema,
        backend="darwin",
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


def test_load_full_label_schema():
    """See test_annotate_external_media"""

    dsn = "fo-v7-test-full-label-schema"
    dataset = fo.load_dataset(dsn)
    for anno_key in ["full_label_schema"]:
        dataset.load_annotations(anno_key)


###########
# cleanup utilities


def list_classes():
    url = f"https://darwin.irl.v7labs.com/api/teams/{team_slug}/annotation_classes"
    headers = {"accept": "application/json", "Authorization": f"ApiKey {api_key}"}
    response = requests.get(url, headers=headers)
    return response.json()


def delete_class(class_id):
    url = f"https://darwin.irl.v7labs.com/api/annotation_classes/{class_id}"
    headers = {"accept": "application/json", "Authorization": f"ApiKey {api_key}"}
    response = requests.delete(url, headers=headers)
    print(response.text)


def list_workflows():
    url = f"https://darwin.irl.v7labs.com/api/v2/teams/{team_slug}/workflows"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"ApiKey {api_key}",
    }
    response = requests.get(url, headers=headers)
    return response.json()


def detach_workflow(workflow):

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"ApiKey {api_key}",
    }

    workflow_id = workflow["id"]
    url = f"https://darwin.irl.v7labs.com/api/v2/teams/{team_slug}/workflows/{workflow_id}/unlink_dataset"
    response = requests.patch(url, headers=headers)
    return response


def delete_workflow(workflow):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"ApiKey {api_key}",
    }

    workflow_id = workflow["id"]
    url = f"https://darwin.irl.v7labs.com/api/v2/teams/{team_slug}/workflows/{workflow_id}"
    response = requests.delete(url, headers=headers)
    print(response.text)


def delete_workflows_unlinked(workflows):
    for workflow in workflows:
        if workflow["dataset"] is None:
            delete_workflow(workflow)


def detach_workflow_dataset(workflows, dataset_name):
    workflow_dsets = {
        x["dataset"]["name"]: x for x in workflows if x["dataset"] is not None
    }
    if dataset_name in workflow_dsets:
        wflow = workflow_dsets[dataset_name]
        response = detach_workflow(wflow)
    else:
        response = None
    return response


def delete_dataset_with_detach(dataset_name):
    config = dfcc.DarwinConfig.from_api_key_with_defaults(api_key)
    client = dfcc.ClientCore(config)
    workflows = dfcw.get_workflows(client, team_slug)
    detach_workflow_dataset(workflows, dataset_name)
    remove_remote_dataset(dataset_name)


def test_annotate_no_dataset_slug(backend_config, setup_quickstart_empty):
    dataset = setup_quickstart_empty
    anno_key = "no_dataset_slug"
    label_field = anno_key
    classes = ["class0", "class1"]
    dataset.annotate(
        anno_key,
        label_field=label_field,
        label_type="classification",
        classes=classes,
        backend="darwin",
        base_url="https://darwin.irl.v7labs.com/api/v2/teams",
    )


@pytest.mark.usefixtures("mocker")
def test_cleanup_all(mocker):
    """Cleanup all datasets, workflows, classes"""
    # Mock user input for cleanup confirmation
    mocker.patch("builtins.input", return_value="y")
    client = Client.from_api_key(api_key)
    api = darwin_fiftyone.darwin.DarwinAPI(api_key)
    dsns0 = list(api._client.list_remote_datasets())
    dsns = [x.name for x in dsns0]
    for dsn in dsns:
        api._delete_dataset_with_workflow_detach(dsn, client)

    wfs = list_workflows()
    delete_workflows_unlinked(wfs)

    cls = list_classes()["annotation_classes"]
    for c in cls:
        delete_class(c["id"])
