import ast
import base64
import json
import logging
import random
import string
import tempfile
import time
import uuid
import webbrowser
import zipfile
from pathlib import Path
import contextlib
import os

import darwin
import darwin.importer as importer
import fiftyone.core.labels as fol
import fiftyone.core.media as fomm
import fiftyone.core.metadata as fom
import fiftyone.utils.annotations as foua
import requests

from darwin.client import Client
from darwin.importer import get_importer

# Set up logging
_DEBUG = True
disable_logging = (
    os.getenv("DISABLE_DARWIN_FIFTYONE_LOGGING", "false").lower() == "true"
)
logging_directory = Path(
    os.getenv("DARWIN_FIFTYONE_LOGGING_DIRECTORY", "~/.fiftyone/v7/")
).expanduser()

if not disable_logging:
    os.makedirs(logging_directory, exist_ok=True)
    log_file_name = (
        logging_directory / f"darwin_fiftyone_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )

    try:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=str(log_file_name),
            filemode="w",
            format="%(asctime)s %(levelname)s %(message)s",
        )
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        file_handler = logging.FileHandler(str(log_file_name))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logging.root.addHandler(file_handler)

    except Exception as e:
        print(f"Failed to write log file to {log_file_name}: {e}")


class DarwinBackendConfig(foua.AnnotationBackendConfig):
    def __init__(
        self,
        name,
        label_schema,
        media_field="filepath",
        api_key=None,
        dataset_slug=None,
        atts=None,
        external_storage=None,
        base_url="https://darwin.v7labs.com/api/v2/teams",
        item_name_annotation=False,
        custom_filename_sample_id_map={},
        Groups=False,
        **kwargs,
    ):
        super().__init__(
            name=name, label_schema=label_schema, media_field=media_field, **kwargs
        )

        if dataset_slug:
            self.dataset_slug = dataset_slug.lower().replace(" ", "-")
        else:
            self.dataset_slug = dataset_slug
        self.atts = atts
        self.external_storage = external_storage
        self.base_url = base_url
        self._api_key = api_key
        self.item_name_annotation = item_name_annotation
        self.custom_filename_sample_id_map = custom_filename_sample_id_map
        self.Groups = Groups

        logging.debug(
            f"Initialized DarwinBackendConfig with dataset_slug: {self.dataset_slug}, base_url: {self.base_url}"
        )

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value
        logging.debug(f"API key set to: {self._api_key}")

    def load_credentials(self, api_key=None):
        self._load_parameters(api_key=api_key)
        logging.debug(f"Loaded credentials with API key: {self._api_key}")


class DarwinBackend(foua.AnnotationBackend):

    @property
    def supported_media_types(self):
        return [fomm.IMAGE, fomm.VIDEO]

    @property
    def supported_label_types(self):
        return [
            "classification",
            "classifications",
            "detection",
            "detections",
            "polygon",
            "polygons",
            "polyline",
            "polylines",
            "keypoint",
            "keypoints",
        ]

    @property
    def supported_attr_types(self):
        return [
            "text",
            "instance_id",
            "single_select",
            "multi_select",
        ]

    @property
    def supports_keyframes(self):
        return True

    @property
    def supports_video_sample_fields(self):
        return False

    @property
    def requires_label_schema(self):
        return True

    def recommend_attr_tool(self, name, value):
        return {"type": "text"}

    def requires_attr_values(self, attr_type):
        return attr_type not in ["text", "instance_id"]

    def upload_annotations(self, samples, anno_key, launch_editor=False):
        logging.debug(
            f"Uploading annotations with anno_key: {anno_key}, samples: {samples}"
        )
        api = self.connect_to_api()
        if anno_key is None:
            anno_key = self._generate_annotation_key()
            logging.info(
                f"No annotation key provided. Generating one with key: {anno_key}"
            )
        results = api.upload_annotations(samples, anno_key, self)

        if launch_editor:
            results.launch_editor()

        return results

    def download_annotations(self, results):
        logging.debug(f"Downloading annotations for results: {results}")
        api = self.connect_to_api()

        logging.info("Downloading labels from V7 Darwin...")
        annotations = api.download_annotations(results)
        logging.info("Download complete")

        return annotations

    def _generate_annotation_key(self):
        alphanumeric_chars = string.ascii_letters + string.digits
        random_string = "".join(random.choice(alphanumeric_chars) for _ in range(15))
        key = random.choice(alphanumeric_chars) + random_string
        logging.debug(f"Generated annotation key: {key}")
        return key

    def _connect_to_api(self):
        logging.debug("Connecting to Darwin API")
        return DarwinAPI(
            api_key=self.config._api_key,
        )


class DarwinAPI(foua.AnnotationAPI):
    def __init__(self, api_key):
        super().__init__()
        self._client = Client.from_api_key(api_key)
        self._api_key = api_key
        logging.debug(f"Initialized DarwinAPI with API key: {self._api_key}")

    def upload_annotations(self, samples, anno_key, backend):
        """
        Uploads annotations to Darwin

        Parameters
        ----------
        samples : fiftyone.core.collections.SampleCollection
            The Voxel51 samples to upload.

        anno_key : str
            The Darwin annotation run key.

        Returns
        -------
        DarwinResults
            The results of the upload.
        """
        logging.info("Uploading annotations")

        label_schema = backend.config.label_schema
        media_field = backend.config.media_field
        external_storage = backend.config.external_storage
        base_url = backend.config.base_url
        custom_filename_sample_id_map = backend.config.custom_filename_sample_id_map
        Groups = backend.config.Groups

        logging.debug(
            f"label_schema: {str(label_schema)}, media_field: {media_field}, external_storage: {external_storage}, base_url: {base_url}, custom_filename_sample_id_map: {custom_filename_sample_id_map}, Groups: {Groups}"
        )

        assert label_schema, "Voxel51 label_schema must be provided"

        if Groups:
            group_dict = {}
            for sample in samples:
                if sample["group"]["id"] not in group_dict:
                    group_dict[sample["group"]["id"]] = [sample]
                else:
                    group_dict[sample["group"]["id"]].append(sample)

        if backend.config.dataset_slug is None:
            backend.config.dataset_slug = f"voxel51-{uuid.uuid4().hex}"

        try:
            dataset = self._client.get_remote_dataset(backend.config.dataset_slug)
        except darwin.exceptions.NotFound:
            dataset = self._client.create_dataset(backend.config.dataset_slug)

        # Single slotted external storage registration
        if external_storage and not Groups:
            logging.info("Registering External Storage Items")
            result = _register_items(
                samples,
                backend.config.api_key,
                dataset.slug,
                dataset.team,
                external_storage,
                base_url,
            )

        # Multislot external storage registration
        elif external_storage and Groups:
            logging.info("Registering Grouped Datasets")
            result = _multislot_registration(
                group_dict,
                backend.config.api_key,
                dataset.slug,
                dataset.team,
                external_storage,
                base_url,
            )

        # Single slotted direct upload
        elif not Groups:
            result = dataset.push(files_to_upload=samples.values(media_field))

        # Multislot direct upload
        else:
            logging.info("Uploading Local Grouped Datasets")
            result = _upload_multislot_items(
                group_dict, backend.config.api_key, dataset.slug, dataset.team, base_url
            )

        filename_sample_id_map = {
            Path(sample[media_field]).name: sample.id for sample in samples
        }

        # Add slot_file_name map. Needs to be sample_name, slot_name, sample_id
        if Groups:
            group_sample_id_map = {}
            for sample in samples:
                if sample["group"]["id"] not in group_sample_id_map:
                    group_sample_id_map[sample["group"]["id"]] = {
                        sample["group"]["name"]: sample.id
                    }
                else:
                    group_sample_id_map[sample["group"]["id"]].update(
                        {sample["group"]["name"]: sample.id}
                    )

        voxel_file_list = (
            [k for k in filename_sample_id_map.keys()]
            if not Groups
            else [k for k in group_sample_id_map.keys()]
        )
        darwin_dataset_items = _list_items(
            backend.config.api_key, dataset.dataset_id, dataset.team, base_url
        )
        annotation_run_items = [
            item for item in darwin_dataset_items if item["name"] in voxel_file_list
        ]

        # Need external storage version and slot logic
        if external_storage and custom_filename_sample_id_map == {} and not Groups:
            item_sample_map = {
                item.get("id"): {
                    "filename": item.get("name"),
                    "sample_slots": {"0": filename_sample_id_map[item.get("name")]},
                }
                for item in annotation_run_items
            }

        # Can be used to handle the duplicate name issue. Doesn't support groups
        elif external_storage and custom_filename_sample_id_map:
            assert (
                not Groups
            ), "Custom filename sample id map not currently supported with groups"
            item_sample_map = {}
            for item in annotation_run_items:
                try:
                    item_sample_map[item.get("id")] = {
                        "filename": item.get("name"),
                        "sample_slots": {
                            "0": custom_filename_sample_id_map[item.get("id")]
                        },
                    }
                except Exception:
                    print(f"Error with item {item.get('id')}. Skipping item.")
                    logging.error(f"Error with item {item.get('id')}. Skipping item.")

        elif custom_filename_sample_id_map and not external_storage and not Groups:
            assert (
                not Groups
            ), "Custom filename sample id map not currently supported with groups"
            item_sample_map = {}
            for item in annotation_run_items:
                try:
                    item_sample_map[item.get("id")] = {
                        "filename": item.get("name"),
                        "sample_slots": {
                            "0": custom_filename_sample_id_map[item.get("id")]
                        },
                    }
                except Exception:
                    print(f"Error with item {item.get('id')}. Skipping item.")
                    logging.error(f"Error with item {item.get('id')}. Skipping item.")

        # Groups external storage
        elif external_storage and Groups:
            item_sample_map = {
                item.get("id"): {
                    "filename": item.get("name"),
                    "sample_slots": group_sample_id_map[item.get("name")],
                }
                for item in annotation_run_items
            }

        # Multislot direct upload
        elif Groups:
            item_sample_map = {
                item.get("id"): {
                    "filename": item.get("name"),
                    "sample_slots": group_sample_id_map[item.get("name")],
                }
                for item in annotation_run_items
            }

        # Single slot direct upload
        else:
            item_sample_map = {
                item.dataset_item_id: {
                    "filename": item.filename,
                    "sample_slots": {"0": filename_sample_id_map[item.filename]},
                }
                for item in result.blocked_items + result.pending_items
            }

        # Wait for all items to be finished processing before attempting to upload annotations
        wait_until_items_finished_processing(
            dataset.dataset_id, dataset.team, backend.config.api_key, base_url
        )

        self._create_missing_annotation_classes(
            backend, dataset.team, label_schema, dataset
        )

        id_maps, frame_id_map = self._upload_annotations(
            label_schema, samples, media_field, dataset, backend
        )

        return DarwinResults(
            samples,
            backend.config,
            anno_key,
            id_maps,
            dataset_slug=dataset.slug,
            item_sample_map=item_sample_map,
            backend=backend,
            frame_id_map=frame_id_map,
        )

    def _convert_image_annotations_to_v7(
        self,
        sample,
        frame_size,
        label_schema,
        id_maps,
        backend,
        slot_name,
        frame_val=None,
    ):
        """
        id_maps: fo sample.id -> list-of-fo-labelobj-ids
        """

        darwin_annotations = []
        for label_field, label_info in label_schema.items():
            try:
                if label_field.startswith("frames"):
                    label_field = label_field.split(".")[1]
                label_type0 = label_info["type"]
                label_type = _UNIQUE_TYPE_MAP.get(label_type0, label_type0)
                if label_type0 == "classification":
                    annotations = [sample[label_field]]
                else:
                    annotations = sample[label_field][label_type]

                logging.info(f"Sample annotations: {annotations}")
                for annotation in annotations:
                    # Adding attributes import support here (depreated attributes field)
                    attributes = getattr(annotation, "attributes", None)
                    if attributes is not None:
                        attribute_list = [
                            str({key: value})
                            for key, value in annotation.attributes.items()
                        ]
                    else:
                        attribute_list = []

                    # Adding directly populated attributes. Most usual way to add attributes. Need to refactor this section
                    if backend.config.atts:
                        for attribute in backend.config.atts:
                            try:
                                attribute_list.append(
                                    str({attribute: annotation[attribute]})
                                )
                            except Exception:
                                logging.error(
                                    f"Error adding attribute {attribute} to annotation {annotation}"
                                )

                    attributes = attribute_list if attribute_list else None

                    darwin_annotations.extend(
                        self._convert_image_annotation_to_v7(
                            annotation,
                            label_type,
                            frame_size,
                            sample,
                            backend,
                            slot_name,
                            attributes,
                        )
                    )

                    if frame_val:
                        if sample.id not in id_maps:
                            id_maps[sample.id] = {}
                            id_maps[sample.id][frame_val.id] = []
                        id_maps[sample.id][frame_val.id].append(annotation.id)

                    else:
                        if sample.id not in id_maps:
                            id_maps[sample.id] = []
                        id_maps[sample.id].append(annotation.id)

            except Exception as e:
                logging.error(f"Error converting image annotations to V7: {e}")

        logging.info(f"Darwin annotations: {darwin_annotations}")
        return darwin_annotations

    def _convert_image_annotation_to_v7(
        self,
        annotation,
        label_type,
        frame_size,
        sample,
        backend,
        slot_name="0",
        attributes=None,
    ):
        """
        Converts a FiftyOne annotation to a Darwin annotation

        Parameters
        ----------
        annotation : fiftyone.core.labels.Label
            The FiftyOne annotation to convert.

        label_type : str
            The type of the label.

        frame_size : tuple
            The size of the frame.

        sample : fiftyone.core.sample.Sample
            The FiftyOne sample to convert.

        Returns
        -------
        list
            A list of Darwin annotations.
        """

        annotation_label = annotation.label

        if backend.config.item_name_annotation and annotation.label in [
            "Item",
            "item",
            "ITEM",
        ]:
            external_path = sample.filepath
            path_list = external_path.split("/")
            sample_name = path_list[-1]
            annotation_label = sample_name

        darwin_annotation = _v7_basic_annotation(
            label=annotation_label,
            confidence=annotation.confidence,
            atts=attributes,
            instance_id=annotation.index,
            slot_name=slot_name,
        )
        logging.info(f"Darwin annotation: {darwin_annotation}")

        if label_type == "detections":
            darwin_annotation["bounding_box"] = _51_to_v7_bbox(
                annotation.bounding_box, frame_size
            )
            logging.info(
                f"Darwin bbox annotation: {darwin_annotation}, input Voxel51 detection: {annotation}"
            )
            return [darwin_annotation]

        if label_type == "classifications":
            darwin_annotation["tag"] = {}
            logging.info(
                f"Darwin tag annotation: {darwin_annotation}, input Voxel51 classification: {annotation}"
            )
            return [darwin_annotation]

        if label_type == "keypoints":
            darwin_annotations = []
            for point in annotation.points:
                darwin_annotation_kp = darwin_annotation.copy()
                darwin_annotation_kp["keypoint"] = _51_to_v7_keypoint(point, frame_size)
                darwin_annotations.append(darwin_annotation_kp)
            logging.info(
                f"Darwin keypoint annotations: {darwin_annotations}, input Voxel51 keypoints: {annotation}"
            )
            return darwin_annotations

        if label_type == "polylines":
            if annotation.closed:
                darwin_annotation["polygon"] = _51_to_v7_polygon(
                    annotation.points, frame_size
                )

            else:
                if len(annotation.points) == 1:
                    darwin_annotation["line"] = _51_to_v7_polyline(
                        annotation.points, frame_size
                    )
                else:
                    logging.info(
                        "Multiple open polylines not supported in V7 Darwin. Please add these as separate annotations."
                    )
            logging.info(
                f"Darwin polyline annotation: {darwin_annotation}, input Voxel51 polyline: {annotation}"
            )
            return [darwin_annotation]

        logging.warn(f"Warning, unsupported label type: {label_type}")

    def _upload_annotations(self, label_schema, samples, media_field, dataset, backend):
        """
        Uploads Voxel annotations to Darwin
        """
        # Go through each sample and upload annotations
        full_id_maps = {}
        frame_id_map = {}
        files_to_import = []

        with (
            contextlib.nullcontext(tempfile.mkdtemp())
            if _DEBUG
            else tempfile.TemporaryDirectory()
        ) as import_path:

            logging.info(f"import_path: {import_path}")
            for label_field, label_info in label_schema.items():
                id_maps = {}
                for sample in samples:

                    slot_name = (
                        sample["group"]["name"] if backend.config.Groups else "0"
                    )

                    logging.info(f"Processing sample: {sample.id}")
                    is_video = sample.media_type == fomm.VIDEO

                    if is_video:
                        assert (
                            sample.frames
                        ), "Video samples must have frames. Please use the ensure_frames() method on your dataset or view to ensure that frames are available."
                        assert label_field.startswith(
                            "frames"
                        ), "Video samples must have frame-level labels"

                    # Checks for videos
                    if sample.metadata is None:
                        if is_video:
                            sample.metadata = fom.VideoMetadata.build_for(
                                sample[media_field]
                            )
                        else:
                            sample.metadata = fom.ImageMetadata.build_for(
                                sample[media_field]
                            )
                    logging.info(f"Sample metadata: {sample.metadata}")

                    if is_video:
                        frame_size = (
                            sample.metadata.frame_width,
                            sample.metadata.frame_height,
                        )
                    else:
                        frame_size = (sample.metadata.width, sample.metadata.height)

                    file_name = Path(sample[media_field]).name
                    darwin_annotations = []

                    if is_video:
                        frames = {}
                        frame_id_map[sample.id] = {}

                        for frame_number, frame in sample.frames.items():
                            frame_id_map[sample.id][str(frame_number)] = frame.id

                            if frame_number not in frames:
                                frames[frame_number] = []

                            frame_val = frame

                            annotations = self._convert_image_annotations_to_v7(
                                frame,
                                frame_size,
                                label_schema,
                                id_maps,
                                backend,
                                slot_name,
                                frame_val,
                            )
                            logging.info(f"Frame annotations: {annotations}")
                            for annotation in annotations:
                                ANNOTATION_DATA_KEYS = [
                                    "bounding_box",
                                    "tag",
                                    "polygon",
                                    "keypoint",
                                ]
                                darwin_frame = {
                                    k: annotation[k]
                                    for k in ANNOTATION_DATA_KEYS
                                    if k in annotation
                                }
                                darwin_frame["keyframe"] = True

                                darwin_annotations.append(
                                    {
                                        "frames": {str(frame_number - 1): darwin_frame},
                                        "name": annotation["name"],
                                        "slot_names": annotation["slot_names"],
                                        "ranges": [[frame_number - 1, frame_number]],
                                    }
                                )
                    else:
                        annotations = self._convert_image_annotations_to_v7(
                            sample,
                            frame_size,
                            label_schema,
                            id_maps,
                            backend,
                            slot_name,
                        )
                        darwin_annotations.extend(annotations)
                    logging.info(f"Sample annotations: {darwin_annotations}")
                    temp_file_path = Path(import_path) / Path(f"{uuid.uuid4()}.json")
                    item_name = (
                        file_name
                        if not backend.config.Groups
                        else sample["group"]["id"]
                    )
                    with open(temp_file_path, "w") as temp_file:
                        json.dump(
                            {
                                "version": "2.0",
                                "item": {"name": item_name, "path": ""},
                                "annotations": darwin_annotations,
                            },
                            temp_file,
                            indent=4,
                        )
                    files_to_import.append(temp_file_path)
                parser = get_importer("darwin")

                if backend.config.Groups:
                    importer.import_annotations(
                        dataset,
                        parser,
                        files_to_import,
                        append=True,
                        class_prompt=False,
                    )
                else:
                    importer.import_annotations(
                        dataset,
                        parser,
                        files_to_import,
                        append=False,
                        class_prompt=False,
                    )

            full_id_maps.update({label_field: id_maps})

        return full_id_maps, frame_id_map

    def _create_missing_annotation_classes(
        self, backend, team_slug, label_schema, dataset
    ):
        """
        Creates and missing annotation classes in V7 Darwin


        Parameters
        ----------
        label_schema : dict
            The Voxel51 label schema.

        dataset : str
            The V7 Darwin dataset to add the annotation classes to.

        Returns
        -------
        str
            A string indicating the success of the operation.
        """

        all_classes = dataset.fetch_remote_classes(team_wide=True)
        # lookup_map = {c["name"]: c for c in all_classes}
        classname_anntype_to_class = {}
        for c in all_classes:
            # class_id = c['id']
            class_name = c["name"]
            for aty in c["annotation_types"]:
                classname_anntype_to_class[(class_name, aty)] = c

        logging.debug(
            f"_create_missing_annotation_classes all_classes:{all_classes} lookup_map:{classname_anntype_to_class} label_schema:{label_schema}"
        )
        for label_field, label_info in label_schema.items():
            classes = label_info["classes"]
            label_type = label_info["type"]
            annotation_type_translation = self.to_darwin_annotation_type(label_type)
            logging.debug(
                f"_create_missing_annotation_classes label_type:{label_type} classes:{classes}"
            )

            classes_to_create = []
            classes_in_team = []
            for cls in classes:
                class_type = type(cls)
                if class_type is str:
                    if (
                        cls,
                        annotation_type_translation,
                    ) not in classname_anntype_to_class:
                        classes_to_create.append(cls)
                    else:
                        classes_in_team.append(cls)
                elif class_type is dict:
                    for subclass in cls["classes"]:
                        if (
                            subclass,
                            annotation_type_translation,
                        ) in classname_anntype_to_class:
                            classes_in_team.append(subclass)
                    cls["classes"] = [
                        c for c in cls["classes"] if c not in classes_in_team
                    ]
                    if cls["classes"]:
                        classes_to_create.append(cls)

            # Only create annotation classes that don't already exist
            for new_cls in classes_to_create:
                class_type = type(new_cls)
                # Create the annotation class if it doesn't exist
                logging.debug(
                    f"Creating class cls:{new_cls} annotation_type_translation:{annotation_type_translation}"
                )
                if class_type is str:
                    subtypes = []
                    classes = [new_cls]
                elif class_type is dict:
                    classes = new_cls["classes"]
                    subtypes = []
                    for attribute in new_cls["attributes"]:
                        if new_cls["attributes"][attribute]["type"] == "instance_id":
                            subtypes.append("instance_id")
                        if new_cls["attributes"][attribute]["type"] == "text":
                            subtypes.append("text")

                for cls in classes:
                    dataset.create_annotation_class(
                        cls, annotation_type_translation, subtypes
                    )

                # If properties then add properties
                if class_type is dict:
                    self._create_properties(backend, team_slug, new_cls, label_type)

            for cls in classes_in_team:
                # if it exists but isn't in the dataset, add it to the dataset
                matching_class = classname_anntype_to_class[
                    (cls, annotation_type_translation)
                ]
                datasets = [dataset["id"] for dataset in matching_class["datasets"]]
                if dataset.dataset_id not in datasets:
                    logging.debug(
                        f"_create_missing_annotation_classes adding to dataset cls:{cls}"
                    )
                    dataset.add_annotation_class(matching_class["id"])

            # Create item level properties
            self._create_item_properties(
                backend, team_slug, label_schema, label_type, dataset.dataset_id
            )

    def _extract_properties(
        self, backend, team_slug, class_schema, prop, label_type, cls=None
    ):
        """
        Extracts properties from a label schema class.

        If `cls` is not provided, the property is assumed to be an item level property.

        Parameters
        ----------
        class_schema : dict
            The class schema to extract properties from.
        prop : str
            The property name to extract

        Returns
        -------
        dict
            The extracted property configuration

        Raises
        ------
        ValueError
            If the property type is not recognized or no type is specified
        """
        is_item_property = cls is None

        property_dict = class_schema["attributes"][prop]
        if "type" not in property_dict:
            raise ValueError(f"No type specified for property {prop}")

        property_type = property_dict["type"]
        vals = property_dict.get("values", [])
        granularity = property_dict.get(
            "granularity",
            "item" if is_item_property else "section"
        )
        required = property_dict.get("required", False)
        cls_id = self._get_annot_class_id(backend, team_slug, cls, label_type)
        payload = {
            "required": required,
            "type": property_type,
            "granularity": granularity,
            "name": prop,
            "description": "",
            "property_values": [],
        }
        if not is_item_property:
            payload["annotation_class_id"] = cls_id

        for val in vals:
            val_dict = {
                "type": "string",
                "color": f"rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},1.0)",
                "value": str(val),
            }
            payload["property_values"].append(val_dict)
        logging.info(f"Extracted properties: {payload}")
        return payload

    def _get_annot_classes(self, backend, team_slug):
        """
        Returns all annotation classes for a given V7 Darwin team.

        This function sends a GET request to the V7 Darwin API and retrieves all annotation classes for a specific team.

        Parameters
        ----------

        team_slug : str
            The slug (unique identifier) of the team in the V7 Darwin platform.

        Returns
        -------
        list
            A list containing all annotation classes in a team.
        """
        base_url = backend.config.base_url
        altered_base_url = base_url.replace("/v2", "")
        url = f"{altered_base_url}/{team_slug}/annotation_classes"

        headers = self._get_headers()

        response = requests.get(url, headers=headers)

        if response.ok:
            return json.loads(response.text)
        else:
            raise requests.exceptions.HTTPError(
                f"GET request failed with status code {response.status_code}."
            )

    def _get_annot_class_id(self, backend, team_slug, name, label_type):
        """
        Retrieves specific class id based on a json response and name

        Parameters
        ----------

        team_slug : str
            The slug (unique identifier) of the team in the V7 Darwin platform.

        name : str
            The name of the class to retrieve

        label_type : str
            The label type of the class

        Returns
        -------
        int
            The id of the class in the V7 Darwin platform.
        """
        darwin_label_type = self.to_darwin_annotation_type(label_type)
        annotation_class_list = self._get_annot_classes(backend, team_slug)
        for an_cls in annotation_class_list["annotation_classes"]:
            if (
                an_cls["name"] == name
                and darwin_label_type in an_cls["annotation_types"]
            ):
                return an_cls["id"]

        return None

    def _create_properties(self, backend, team_slug, class_schema, label_type):
        """
        Creates properties for a given class in a label_schema. If the property already exists, it will be updated with
        additional values.

        Parameters
        ----------
        team_slug : str
            The slug (unique identifier) of the team in the V7 Darwin platform.

        class_schema : dict
            The class schema to extract properties from.

        label_type : str
            The label type of the class

        Returns
        -------
        str
            A string indicating the success of the operation.
        """
        base_url = backend.config.base_url
        url = f"{base_url}/{team_slug}/properties"
        headers = self._get_headers()

        for prop in class_schema["attributes"].keys():
            if "text" in class_schema["attributes"][prop]["type"]:
                logging.info(
                    "Text attribute not supported in V7 Darwin. Please use the text subattribute instead."
                )
                continue
            if "instance_id" in class_schema["attributes"][prop]["type"]:
                logging.info(
                    "Instance ID subtype cannot be associated with values. Continuing."
                )
                continue

            for cls in class_schema["classes"]:
                payload = self._extract_properties(
                    backend, team_slug, class_schema, prop, label_type, cls
                )
                cls_id = payload["annotation_class_id"]
                properties_check = self._check_properties(
                    backend, team_slug, cls_id, prop
                )
                if not properties_check:
                    response = requests.post(url, json=payload, headers=headers)
                    if not response.ok:
                        raise requests.exceptions.HTTPError(
                            f"POST request failed with status code {response.status_code}."
                        )
                else:
                    current_property_values = [
                        val["value"] for val in properties_check[0]["property_values"]
                    ]
                    payload_property_values = [
                        val["value"] for val in payload["property_values"]
                    ]
                    if any(
                        val not in current_property_values
                        for val in payload_property_values
                    ):
                        prop_id = self._get_property_id(properties_check, prop)
                        url = f"{base_url}/{team_slug}/properties/{prop_id}"
                        del payload["granularity"]
                        response = requests.put(url, json=payload, headers=headers)
                        if not response.ok:
                            raise requests.exceptions.HTTPError(
                                f"PUT request failed with status code {response.status_code}."
                            )

    def _create_item_properties(
        self, backend, team_slug, label_schema, label_type, dataset_id
    ):
        """
        Creates item level properties for a given label_schema.
        """
        base_url = backend.config.base_url
        headers = self._get_headers()

        for label_field, label_info in label_schema.items():
            for item_property_name in label_info["attributes"]:

                if "instance_id" in label_info["attributes"][item_property_name]["type"]:
                    # instance_id is not to be considered as item property
                    continue
                payload = self._extract_properties(
                    backend,
                    team_slug,
                    label_schema[label_field],
                    item_property_name,
                    label_type,
                )
                properties_check = self._check_item_properties(
                    backend, team_slug, item_property_name
                )
                if not properties_check:
                    url = f"{base_url}/{team_slug}/properties"
                    payload["dataset_ids"] = [dataset_id]
                    response = requests.post(url, json=payload, headers=headers)
                    if not response.ok:
                        raise requests.exceptions.HTTPError(
                            f"POST request failed with status code {response.status_code}."
                        )
                    continue
                else:
                    current_property_values = [
                        val["value"] for val in properties_check[0]["property_values"]
                    ]
                    payload_property_values = [
                        val["value"] for val in payload["property_values"]
                    ]
                    if any(
                        val not in current_property_values
                        for val in payload_property_values
                    ):
                        prop_id = self._get_property_id(
                            properties_check, item_property_name
                        )
                        url = f"{base_url}/{team_slug}/properties/{prop_id}"
                        del payload["granularity"]
                        response = requests.put(url, json=payload, headers=headers)
                        if not response.ok:
                            raise requests.exceptions.HTTPError(
                                f"PUT request failed with status code {response.status_code}."
                            )
                    # If needed, assign the item property to the dataset
                    if dataset_id not in properties_check[0]["dataset_ids"]:
                        prop_id = self._get_property_id(
                            properties_check, item_property_name
                        )
                        dataset_ids = properties_check[0]["dataset_ids"] + [dataset_id]
                        # The below is a workaround for DAR-5770
                        # It should be removed once the issue is resolved
                        current_datasets = self._get_datasets(backend, team_slug)
                        current_dataset_ids = [
                            dataset["id"] for dataset in current_datasets
                        ]
                        dataset_ids = [
                            dataset_id
                            for dataset_id in dataset_ids
                            if dataset_id in current_dataset_ids
                        ]
                        url = f"{base_url}/{team_slug}/properties/{prop_id}"
                        response = requests.put(
                            url,
                            json={
                                "dataset_ids": dataset_ids,
                                "name": item_property_name,
                            },
                            headers=headers,
                        )
                        if not response.ok:
                            raise requests.exceptions.HTTPError(
                                f"PUT request failed with status code {response.status_code}."
                            )

    def _check_item_properties(self, backend, team_slug, item_property_name):
        """
        Check for the existence of a given item property
        """
        base_url = backend.config.base_url
        url = f"{base_url}/{team_slug}/properties?include_values=true"
        headers = self._get_headers()
        response = requests.get(url, headers=headers)
        item_properties = json.loads(response.text)["properties"]
        for prop in item_properties:
            if prop["name"] == item_property_name:
                return [prop]
        return []

    def _check_properties(self, backend, team_slug, class_id, prop_name):
        """
        Checks for the existence of a given property in a class

        Parameters
        ----------
        team_slug : str
            The slug (unique identifier) of the team in the V7 Darwin platform.

        class_id : int
            The id of the class in the V7 Darwin platform.

        Returns
        -------
        dict
            A dictionary containing all properties in a class.
        """
        base_url = backend.config.base_url
        url = f"{base_url}/{team_slug}/properties?annotation_class_ids[]={class_id}&include_values=true"
        headers = self._get_headers()
        response = requests.get(url, headers=headers)
        properties = json.loads(response.text)["properties"]
        for prop in properties:
            if prop["name"] == prop_name:
                return [prop]
        return []

    def _get_property_id(self, properties, name):
        """
        Retrieves specific property id based on a json response and name

        Parameters
        ----------
        properties : dict
            The properties to retrieve the id from

        Returns
        -------
        int
            The id of the property in the V7 Darwin platform.
        """
        for prop in properties:
            if prop["name"] == name:
                return prop["id"]

        return None

    def to_darwin_annotation_type(self, type):
        if type == "detections" or type == "detection":
            return "bounding_box"
        if type == "classification" or type == "classifications":
            return "tag"
        if type == "keypoints" or type == "keypoint":
            return "keypoint"
        if type == "polylines" or type == "polygon" or type == "polygons":
            return "polygon"
        if type == "polyline":
            return "line"
        raise ValueError(f"Unknown type {type}")

    def download_annotations(self, results):
        """
        Downloads annotations from V7 Darwin

        Parameters
        ----------
        results : DarwinResults
            The results of the upload.

        Returns
        -------
        dict
            The downloaded annotations.
        """
        label_schema = results.config.label_schema
        item_sample_map = results.item_sample_map
        frame_id_map = results.frame_id_map
        item_name_annotation = results.config.item_name_annotation
        logging.info(f"results: {results}")

        dataset = self._client.get_remote_dataset(results.config.dataset_slug)
        full_annotations = {}
        with (
            contextlib.nullcontext(tempfile.mkdtemp())
            if _DEBUG
            else tempfile.TemporaryDirectory()
        ) as release_path:

            logging.info(f"release_path: {release_path}")

            export_path = self._generate_export(release_path, dataset)
            for label_field, label_info in label_schema.items():

                label_type = label_info["type"]
                label_type = _UNIQUE_TYPE_MAP.get(label_type, label_type)
                sample_annotations = {}

                for annotation_path in export_path.glob("*.json"):
                    data = json.loads(annotation_path.read_text(encoding="UTF-8"))
                    item_id = data["item"]["source_info"]["item_id"]
                    if item_id not in item_sample_map:
                        logging.warn(
                            f"WARNING, {item_id} not in item_sample_map, skipping"
                        )
                        continue

                    widths, heights = {}, {}
                    for k in data["item"]["slots"]:
                        widths.update({k["slot_name"]: k["width"]})
                        heights.update({k["slot_name"]: k["height"]})

                    item_name = data["item"]["name"]

                    is_darwin_video = False

                    if data["item"]["slots"][0]["type"] == "video":
                        is_darwin_video = True
                        frame_count = data["item"]["slots"][0]["frame_count"]
                        frame_dict = {}
                        for frame_number in range(1, frame_count + 2):
                            frame_dict.update({frame_number: {}})

                    annotations = {}

                    # Video annotations
                    if is_darwin_video:
                        logging.info("Processing video annotations")

                        # Spits merged annotations into per frame annotations
                        video_annotations = self._split_video_annotations(
                            data["annotations"]
                        )
                        logging.info(f"Video annotations: {video_annotations}")

                        for annotation in video_annotations:
                            annot_name = annotation["name"]
                            slot_name = annotation["slot_names"][0]
                            sample_id = item_sample_map[item_id]["sample_slots"][
                                slot_name
                            ]
                            width, height = widths[slot_name], heights[slot_name]
                            frame_number = list(annotation["frames"].keys())[0]
                            frame_annotation = annotation["frames"][frame_number]
                            confidence = None

                            if item_name_annotation:
                                annot_name = (
                                    annotation["name"]
                                    if annotation["name"] != item_name
                                    else "Item"
                                )

                            # Checks for unsupported annotation types
                            if (
                                "polygon" not in frame_annotation
                                and "bounding_box" not in frame_annotation
                                and "tag" not in frame_annotation
                                and "keypoint" not in frame_annotation
                                and "line" not in frame_annotation
                            ):
                                logging.warn(
                                    "WARNING, unsupported annotation type", annotation
                                )
                                continue

                            darwin_attributes = {}
                            direct_attribute = False
                            if "attributes" in frame_annotation:
                                direct_attribute_dict = {}
                                for attribute in frame_annotation["attributes"]:
                                    # Searching to see if an original attribute or newly added in V7
                                    if "{" in attribute:
                                        direct_attribute = True
                                        attribute = ast.literal_eval(attribute)
                                        direct_attribute_dict.update(attribute)
                                    else:
                                        darwin_attributes[attribute] = True

                            if "properties" in annotation:
                                for prop in annotation["properties"]:
                                    assert (
                                        prop["name"] != "Text"
                                    ), "Text is a reserved name and cannot be used as a property name"
                                    if prop["name"] in darwin_attributes:
                                        if isinstance(
                                            darwin_attributes[prop["name"]], list
                                        ):
                                            darwin_attributes[prop["name"]].append(
                                                prop["value"]
                                            )
                                        else:
                                            tmp_list = [
                                                darwin_attributes[prop["name"]],
                                                prop["value"],
                                            ]
                                            darwin_attributes[prop["name"]] = tmp_list
                                    else:
                                        darwin_attributes[prop["name"]] = prop["value"]

                            # Add darwin annotators
                            if "annotators" in annotation and annotation["annotators"]:
                                darwin_attributes["darwin_annotators"] = []
                                for annotator in annotation["annotators"]:
                                    darwin_attributes["darwin_annotators"].append(
                                        annotator["full_name"]
                                    )

                            # Add darwin reviewers
                            if "reviewers" in annotation and annotation["reviewers"]:
                                darwin_attributes["darwin_reviewers"] = []
                                for reviewer in annotation["reviewers"]:
                                    darwin_attributes["darwin_reviewers"].append(
                                        reviewer["full_name"]
                                    )

                            # Add instance id
                            if "instance_id" in frame_annotation:
                                darwin_attributes["darwin_instance_id"] = (
                                    frame_annotation["instance_id"]["value"]
                                )

                            # Adding text support
                            if "text" in frame_annotation:
                                darwin_attributes["Text"] = frame_annotation["text"][
                                    "text"
                                ]

                            # Add in/out view or occluded here
                            if "hidden_areas" in annotation:
                                darwin_attributes["occluded"] = "False"
                                for hidden_area in annotation["hidden_areas"]:
                                    if (
                                        hidden_area[0]
                                        <= int(frame_number)
                                        <= hidden_area[1]
                                    ):
                                        darwin_attributes["occluded"] = "True"

                            if "inference" in frame_annotation:
                                confidence = (
                                    frame_annotation["inference"]["confidence"]
                                    if "confidence" in frame_annotation["inference"]
                                    else None
                                )

                            voxel_annotation = None
                            if "polygon" in frame_annotation and label_type in [
                                "polygons",
                                "polylines",
                            ]:
                                if len(frame_annotation["polygon"]["paths"]) == 1:
                                    logging.info("Processing simple polygons")
                                    voxel_annotation = _v7_to_51_polygon(
                                        annot_name,
                                        frame_annotation["polygon"],
                                        height,
                                        width,
                                        attributes=darwin_attributes,
                                    )
                                else:
                                    logging.info("Processing complex polygons")
                                    voxel_annotation = _v7_to_51_polygon(
                                        annot_name,
                                        frame_annotation["polygon"],
                                        height,
                                        width,
                                        attributes=darwin_attributes,
                                    )

                            elif "line" in frame_annotation and label_type in [
                                "polyline",
                                "polylines",
                            ]:
                                voxel_annotation = _v7_to_51_polyline(
                                    annot_name,
                                    frame_annotation["line"],
                                    height,
                                    width,
                                    attributes=darwin_attributes,
                                )

                            elif (
                                "bounding_box" in frame_annotation
                                and "polygon" not in frame_annotation
                                and label_type in ["detections", "detection"]
                            ):
                                voxel_annotation = _v7_to_51_bbox(
                                    annot_name,
                                    frame_annotation["bounding_box"],
                                    height,
                                    width,
                                    attributes=darwin_attributes,
                                )
                            elif "tag" in frame_annotation and label_type in [
                                "classifications",
                                "classification",
                            ]:
                                voxel_annotation = _v7_to_51_classification(
                                    annot_name,
                                    attributes=darwin_attributes,
                                )
                            elif "keypoint" in frame_annotation and label_type in [
                                "keypoints",
                                "keypoint",
                            ]:
                                voxel_annotation = _v7_to_51_keypoint(
                                    annot_name,
                                    frame_annotation["keypoint"],
                                    height,
                                    width,
                                    attributes=darwin_attributes,
                                )

                            # Ignores non matching annotation types with label_type
                            if not voxel_annotation:
                                continue

                            # Adding confidence score
                            if confidence:
                                voxel_annotation.confidence = confidence

                            # Adding direct attributes
                            if direct_attribute:
                                for key, val in direct_attribute_dict.items():
                                    voxel_annotation[key] = val

                            if voxel_annotation.id in annotations:
                                if type(voxel_annotation) is fol.Keypoint:
                                    annotations[voxel_annotation.id].points.extend(
                                        voxel_annotation.points
                                    )

                            # Need to add condition for adding both bbox, and polygon
                            else:
                                new_num = str(int(frame_number) + 1)

                                if new_num in frame_id_map[sample_id].keys():
                                    frame_id = frame_id_map[sample_id][new_num]
                                    if frame_id not in annotations.keys():
                                        annotations[frame_id] = {
                                            voxel_annotation.id: voxel_annotation
                                        }

                                    else:
                                        annotations[frame_id].update(
                                            {voxel_annotation.id: voxel_annotation}
                                        )

                                else:
                                    frame_id = None
                                    annotations[frame_id] = {
                                        voxel_annotation.id: voxel_annotation
                                    }

                            # Check this section for keypoints. Need to adjust either way. Think this was for multislot
                            if sample_id not in sample_annotations:
                                sample_annotations[sample_id] = {}

                            if sample_id in annotations:
                                if frame_id not in sample_annotations[sample_id]:
                                    sample_annotations[sample_id][frame_id] = {}
                                sample_annotations[sample_id][frame_id].update(
                                    annotations[frame_id]
                                )
                            else:
                                if frame_id not in sample_annotations[sample_id]:
                                    sample_annotations[sample_id][frame_id] = {}
                                sample_annotations[sample_id][frame_id].update(
                                    annotations[frame_id]
                                )

                        # sample_annotations[sample_id] = annotations

                    # Image annotations
                    else:
                        logging.info("Processing image annotations")
                        for annotation in data["annotations"]:
                            annot_name = annotation["name"]
                            slot_name = annotation["slot_names"][0]
                            sample_id = item_sample_map[item_id]["sample_slots"][
                                slot_name
                            ]
                            width, height = widths[slot_name], heights[slot_name]
                            confidence = None

                            if item_name_annotation:
                                annot_name = (
                                    annotation["name"]
                                    if annotation["name"] != item_name
                                    else "Item"
                                )

                            # Checks for unsupported annotation types
                            if (
                                "polygon" not in annotation
                                and "bounding_box" not in annotation
                                and "tag" not in annotation
                                and "keypoint" not in annotation
                                and "line" not in annotation
                            ):
                                logging.warn(
                                    "WARNING, unsupported annotation type", annotation
                                )
                                continue

                            direct_attribute = False
                            darwin_attributes = {}
                            if "attributes" in annotation:
                                direct_attribute_dict = {}
                                for attribute in annotation["attributes"]:
                                    # Searching to see if an original attribute or newly added in V7
                                    if "{" in attribute:
                                        direct_attribute = True
                                        attribute = ast.literal_eval(attribute)
                                        direct_attribute_dict.update(attribute)
                                    else:
                                        darwin_attributes[attribute] = True

                            if "properties" in annotation:
                                for prop in annotation["properties"]:
                                    assert (
                                        prop["name"] != "Text"
                                    ), "Text is a reserved name and cannot be used as a property name"
                                    if prop["name"] in darwin_attributes:
                                        if isinstance(
                                            darwin_attributes[prop["name"]], list
                                        ):
                                            darwin_attributes[prop["name"]].append(
                                                prop["value"]
                                            )
                                        else:
                                            tmp_list = [
                                                darwin_attributes[prop["name"]],
                                                prop["value"],
                                            ]
                                            darwin_attributes[prop["name"]] = tmp_list
                                    else:
                                        darwin_attributes[prop["name"]] = prop["value"]

                            # Add darwin annotators
                            if "annotators" in annotation and annotation["annotators"]:
                                darwin_attributes["darwin_annotators"] = []
                                for annotator in annotation["annotators"]:
                                    darwin_attributes["darwin_annotators"].append(
                                        annotator["full_name"]
                                    )

                            # Add darwin reviewers
                            if "reviewers" in annotation and annotation["reviewers"]:
                                darwin_attributes["darwin_reviewers"] = []
                                for reviewer in annotation["reviewers"]:
                                    darwin_attributes["darwin_reviewers"].append(
                                        reviewer["full_name"]
                                    )

                            # Add instance id
                            if "instance_id" in annotation:
                                darwin_attributes["darwin_instance_id"] = annotation[
                                    "instance_id"
                                ]["value"]

                            # Adding text support
                            if "text" in annotation:
                                darwin_attributes["Text"] = annotation["text"]["text"]

                            if "inference" in annotation:
                                confidence = (
                                    annotation["inference"]["confidence"]
                                    if "confidence" in annotation["inference"]
                                    else None
                                )

                            voxel_annotation = None
                            if "polygon" in annotation and label_type in [
                                "polygons",
                                "polygon",
                                "polylines",
                            ]:
                                if len(annotation["polygon"]["paths"]) == 1:
                                    logging.info("Processing simple polygons")
                                    voxel_annotation = _v7_to_51_polygon(
                                        annot_name,
                                        annotation["polygon"],
                                        height,
                                        width,
                                        attributes=darwin_attributes,
                                    )
                                else:
                                    logging.info("Processing complex polygons")
                                    voxel_annotation = _v7_to_51_polygon(
                                        annot_name,
                                        annotation["polygon"],
                                        height,
                                        width,
                                        attributes=darwin_attributes,
                                    )

                            elif "line" in annotation and label_type in [
                                "polyline",
                                "polylines",
                            ]:
                                voxel_annotation = _v7_to_51_polyline(
                                    annot_name,
                                    annotation["line"],
                                    height,
                                    width,
                                    attributes=darwin_attributes,
                                )

                            elif (
                                "bounding_box" in annotation
                                and "polygon" not in annotation
                                and label_type in ["detections", "detection"]
                            ):
                                voxel_annotation = _v7_to_51_bbox(
                                    annot_name,
                                    annotation["bounding_box"],
                                    height,
                                    width,
                                    attributes=darwin_attributes,
                                )
                            elif "tag" in annotation and label_type in [
                                "classifications",
                                "classification",
                            ]:
                                voxel_annotation = _v7_to_51_classification(
                                    annot_name,
                                    attributes=darwin_attributes,
                                )
                            elif "keypoint" in annotation and label_type in [
                                "keypoints",
                                "keypoint",
                            ]:
                                voxel_annotation = _v7_to_51_keypoint(
                                    annot_name,
                                    annotation["keypoint"],
                                    height,
                                    width,
                                    attributes=darwin_attributes,
                                )

                            # Ignores non matching annotation types with label_type
                            if not voxel_annotation:
                                continue

                            # Adding confidence score
                            if confidence:
                                voxel_annotation.confidence = confidence

                            # Adding direct attributes
                            if direct_attribute:
                                for key, val in direct_attribute_dict.items():
                                    voxel_annotation[key] = val

                            if voxel_annotation.id in annotations:
                                if type(voxel_annotation) is fol.Keypoint:
                                    annotations[voxel_annotation.id].points.extend(
                                        voxel_annotation.points
                                    )

                            # Need to add condition for adding both bbox, and polygon
                            else:
                                annotations[voxel_annotation.id] = voxel_annotation

                            # Update for annotations so works with slots
                            if sample_id in sample_annotations:
                                sample_annotations[sample_id].update(
                                    {voxel_annotation.id: voxel_annotation}
                                )

                            else:
                                sample_annotations[sample_id] = {
                                    voxel_annotation.id: voxel_annotation
                                }

                full_annotations.update({label_field: {label_type: sample_annotations}})

        logging.info(f"Full annotations: {full_annotations}")
        return full_annotations

    def _generate_export(self, release_path, dataset):
        """
        Generates a V7 export file
        """
        release_name: str = f"voxel51-{uuid.uuid4().hex}"
        self._client.api_v2.export_dataset(
            format="darwin_json_2",
            name=release_name,
            include_authorship=True,
            include_token=False,
            annotation_class_ids=None,
            filters={"not_statuses": ["archived", "error"]},
            dataset_slug=dataset.slug,
            team_slug=dataset.team,
        )

        logging.info("Create export ")
        backoff = 1
        zipfile_path = Path(release_path) / Path(f"{release_name}.zip")
        extracted_path = Path(release_path) / Path(f"{release_name}")
        while True:
            time.sleep(backoff)
            logging.info(".")
            try:
                release = dataset.get_release(release_name, include_unavailable=False)
                release.download_zip(zipfile_path)

                with zipfile.ZipFile(zipfile_path, "r") as zip:
                    zip.extractall(extracted_path)
                return extracted_path

            except darwin.exceptions.NotFound:
                backoff += 1
                continue

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"ApiKey {self._api_key}",
        }

    def _get_workflows_url(self, team_slug, base_url):
        try:
            return f"{base_url}/{team_slug}/workflows"
        except darwin.exceptions.NotFound:
            raise darwin.exceptions.NotFound(f"Team {team_slug} not found")

    def _get_workflows(self, team_slug, base_url):
        url = self._get_workflows_url(team_slug, base_url)
        headers = self._get_headers()
        response = requests.get(url, headers=headers)
        if response.ok:
            return json.loads(response.text)
        else:
            raise requests.exceptions.HTTPError(
                f"GET request failed with status code {response.status_code}."
            )

    def _detach_workflow(self, team_slug, workflow, base_url):
        url = self._get_workflows_url(team_slug, base_url)
        headers = self._get_headers()
        workflow_id = workflow["id"]
        url = f"{url}/{workflow_id}/unlink_dataset"
        response = requests.patch(url, headers=headers)
        logging.info("Unlinking workflow {workflow_id}")
        if response.ok:
            logging.info(f"Workflow {workflow_id} unlinked")
            return response.json()
        else:
            raise requests.exceptions.HTTPError(
                f"PATCH request failed with status code {response.status_code}."
            )

    def _get_datasets(self, backend, team_slug):
        url = "/".join(backend.config.base_url.split("/")[0:4]) + "/datasets"
        headers = self._get_headers()
        response = requests.get(url, headers=headers)
        return json.loads(response.text)

    def _delete_dataset_id(self, dataset_id, base_url):
        url = f"{base_url}/datasets/{dataset_id}/archive"
        headers = self._get_headers()
        response = requests.put(url, headers=headers)
        if response.ok:
            logging.info(f"Dataset {dataset_id} deleted")
            return response.json()
        else:
            raise requests.exceptions.HTTPError(
                f"PUT request failed with status code {response.status_code}."
            )

    def _delete_workflow_id(self, team_slug, workflow_id, base_url):
        url = f"{base_url}/{team_slug}/workflows/{workflow_id}"
        headers = self._get_headers()
        response = requests.delete(url, headers=headers)
        if response.ok:
            logging.info(f"Workflow {workflow_id} deleted")
            return response.json()
        else:
            raise requests.exceptions.HTTPError(
                f"DELETE request failed with status code {response.status_code}."
            )

    def _delete_dataset_with_workflow_detach(self, dataset_slug, client):
        dataset = self._client.get_remote_dataset(dataset_slug)
        base_url = client.base_url + "/api/v2/teams"
        dataset_name = dataset.name
        team_slug = dataset.team
        workflows = self._get_workflows(team_slug, base_url)
        workflow_dsets = {
            x["dataset"]["name"]: x for x in workflows if x["dataset"] is not None
        }
        if dataset_name in workflow_dsets:
            wflow = workflow_dsets[dataset_name]
            self._detach_workflow(team_slug, wflow, base_url)
            logging.info(f"Detaching workflow for dataset {dataset_name}")
            self._delete_workflow_id(team_slug, wflow["id"], base_url)
        else:
            logging.warning(f"Did not find workflow for dataset {dataset_name}")

        dataset_base_url = client.base_url + "/api"
        self._delete_dataset_id(dataset.dataset_id, dataset_base_url)
        logging.info(f"Deleting dataset {dataset_name}")

    def _split_video_annotations(self, annotations):
        """
        Splits Darwin JSON video annotations into per frame annotations
        """
        converted_annotations = []

        for annot in annotations:
            frame_dict = annot.pop("frames")

            for key in frame_dict.keys():
                new_annot = {}
                new_annot.update(annot)
                new_annot["frames"] = {key: frame_dict[key]}
                new_annot["ranges"] = [[int(key), int(key) + 1]]
                converted_annotations.append(new_annot)

        return converted_annotations

    @property
    def client(self):
        return self._client


class DarwinResults(foua.AnnotationResults):
    def __init__(
        self,
        samples,
        config,
        anno_key,
        id_map,
        item_sample_map=None,
        dataset_slug=None,
        backend=None,
        attributes=None,
        frame_id_map=None,
    ):
        super().__init__(samples, config, anno_key, id_map, backend=backend)
        self.dataset_slug = dataset_slug
        self.item_sample_map = item_sample_map
        self.atts = attributes
        self.frame_id_map = frame_id_map
        self.id_map = id_map
        self.anno_key = anno_key

    def launch_editor(self):
        """
        Launches the V7 Darwin tool
        """
        client = self.connect_to_api().client
        dataset = client.get_remote_dataset(self.backend.config.dataset_slug)
        url = f"{client.base_url}/datasets/{dataset.dataset_id}"
        webbrowser.open(url, new=2)

    def cleanup(self):
        """
        Cleans up annotations in V7 Darwin by deleting annotations and returning items to new status
        """
        api = self.connect_to_api()
        client = api.client
        answer = input(
            "Are you sure you want to delete the V7 dataset and workflow for this annotation run? (y/n)"
        )
        if answer.lower() == "y":
            api._delete_dataset_with_workflow_detach(self.dataset_slug, client)
        else:
            logging.info("Cleanup cancelled")

    def check_status(self):
        """
        Checks and prints the status of annotations in V7 Darwin.
        """
        client = self.connect_to_api().client
        dataset = client.get_remote_dataset(self.backend.config.dataset_slug)
        url = f"{client.base_url}/api/v2/teams/{dataset.team}/items/status_counts?dataset_ids[]={dataset.dataset_id}"
        response = requests.get(url, headers=self._get_headers())

        response.raise_for_status()

        annotation_statuses = []
        print(f"Annotation Run Status Counts for {self.anno_key}:")
        for status_obj in json.loads(response.text)["simple_counts"]:
            print(f"{status_obj['status']}: {status_obj['item_count']}")
            annotation_statuses.append(status_obj["status"])
        return annotation_statuses

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"ApiKey {self.backend.config.api_key}",
        }

    @classmethod
    def _from_dict(cls, d, samples, config, anno_key):
        return cls(
            samples,
            config,
            anno_key,
            id_map=d["id_map"],
            item_sample_map=d.get("item_sample_map"),
            dataset_slug=d.get("dataset_slug"),
            frame_id_map=d.get("frame_id_map"),
        )


# Registering External Storage Items
def _register_items(
    samples, api_key, dataset_slug, team_slug, external_storage, base_url
):
    """
    Registers external storage items in Darwin

    Only readwrite currently supported
    """
    logging.info("item registration started")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"ApiKey {api_key}",
    }

    item_list = []

    for sample in samples:
        external_path = sample.filepath
        path_list = external_path.split("/")
        name = path_list[-1]
        item_list.append(name)
        new_path = "/".join(path_list[3:])
        temp_dict = {
            "path": "/",
            "slots": [
                {
                    "as_frames": "false",
                    "slot_name": "0",
                    "storage_key": new_path,
                    "file_name": name,
                }
            ],
            "name": name,
        }

        payload = {
            "items": [temp_dict],
            "dataset_slug": dataset_slug,
            "storage_slug": external_storage,
        }
        logging.info(f"payload: {payload}")

        for chunk in _chunk_list(payload["items"], 10):
            chunked_payload = payload.copy()
            chunked_payload["items"] = chunk
            backoff = 1
            while True:
                response = requests.post(
                    f"{base_url}/{team_slug}/items/register_existing",
                    headers=headers,
                    json=chunked_payload,
                )

                if response.ok:
                    logging.info(f"Item registration complete for {name}")
                    break
                elif response.status_code == 429:
                    logging.warning(
                        f"Rate limit exceeded. Retrying in {backoff} seconds..."
                    )
                    time.sleep(backoff)
                    backoff = min(
                        backoff * 2, 300
                    )  # Exponential backoff with a max delay
                else:
                    logging.error(
                        f"Item {name} registration failed with status code {response.status_code}. Skipping item registration"
                    )
                    break

    return item_list


# Multislot item registration
def _multislot_registration(
    groups, api_key, dataset_slug, team_slug, external_storage, base_url
):
    """
    Registers external storage items in Darwinin multislots defined by groups

    Only readwrite currently supported
    """
    logging.info("Multi slot item registration started")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"ApiKey {api_key}",
    }

    item_list = []

    for k in groups:
        group = groups[k]
        temp_dict = {
            "path": "/",
            "slots": [],
            "name": group[0]["group"]["id"],
        }
        for sample in group:
            external_path = sample.filepath
            path_list = external_path.split("/")
            name = path_list[-1]
            item_list.append(name)
            new_path = "/".join(path_list[3:])
            group_id = sample.group.id
            if sample.media_type in ["image", "video"]:
                temp_dict["slots"].append(
                    {
                        "as_frames": "false",
                        "slot_name": str(sample["group"]["name"]),
                        "storage_key": new_path,
                        "file_name": name,
                    }
                )
            else:
                logging.warning(
                    f"Media type {sample.media_type} not supported for multi slot item registration. Skipping item registration"
                )

        payload = {
            "items": [temp_dict],
            "dataset_slug": dataset_slug,
            "storage_slug": external_storage,
        }
        logging.info(f"payload: {payload}")

        for chunk in _chunk_list(payload["items"], 10):
            chunked_payload = payload.copy()
            chunked_payload["items"] = chunk
            backoff = 1
            while True:
                response = requests.post(
                    f"{base_url}/{team_slug}/items/register_existing",
                    headers=headers,
                    json=chunked_payload,
                )

                if response.ok:
                    logging.info(f"Multi slot item registration complete for {name}")
                    break
                elif response.status_code == 429:
                    logging.warning(
                        f"Rate limit exceeded. Retrying in {backoff} seconds..."
                    )
                    time.sleep(backoff)
                    backoff = min(
                        backoff * 2, 300
                    )  # Exponential backoff with a max delay
                else:
                    logging.error(
                        f"Multi slot item with id {group_id} registration failed with status code {response.status_code}. Skipping item registration"
                    )
                    break

    return item_list


# Uploads local group datasets
def _upload_multislot_items(groups, api_key, dataset_slug, team_slug, base_url):
    """
    Uploads local group datasets to V7 Darwin
    """
    item_list = []
    for k in groups:
        group = groups[k]
        group_id = group[0]["group"]["id"]
        temp_dict = {
            "path": "/",
            "slots": [],
            "name": group_id,
        }

        for sample in group:
            local_path = sample.filepath
            path_list = local_path.split("/")
            name = path_list[-1]
            item_list.append(name)
            image = open(local_path, "rb")
            image_b64 = base64.b64encode(image.read()).decode("ascii")
            if sample.media_type in ["image", "video"]:
                temp_dict["slots"].append(
                    {
                        "file_content": image_b64,
                        "slot_name": str(sample["group"]["name"]),
                        "file_name": name,
                    }
                )
            else:
                logging.warning(
                    f"Media type {sample.media_type} not supported for multi slot item registration. Skipping item registration"
                )

        payload = {
            "items": [temp_dict],
            "dataset_slug": dataset_slug,
        }

        response = requests.post(
            f"{base_url}/{team_slug}/items/direct_upload",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"ApiKey {api_key}",
            },
            json=payload,
        )

        if response.ok:
            logging.info(f"Multi slot item upload complete for group {group_id}")
        else:
            logging.error(
                f"Multi slot item with id {group_id} upload failed with status code {response.status_code}. Skipping item upload"
            )

    return item_list


def _v7_basic_annotation(
    label,
    annotators: list = [],
    reviewers: list = [],
    confidence=None,
    atts: list = [],
    text: str = "",
    instance_id: str = "",
    slot_name="0",
):
    """
    Creates a base V7 annotation
    """
    annot = {}
    annot_list = []
    if annotators:
        for annotator in annotators:
            new_annot = {}
            new_annot["email"] = annotator["email"]
            new_annot["full_name"] = annotator["full_name"]
            annot_list.append(new_annot)
    annot["annotators"] = annot_list

    if atts:
        annot["attributes"] = atts

    if confidence:
        model = {"id": str(uuid.uuid4()), "name": "Voxel51", "type": "external"}
        annot["inference"] = {"confidence": confidence, "model": model}

    if instance_id:
        annot["instance_id"] = {"value": instance_id}

    annot["name"] = label

    annot["slot_names"] = [slot_name]

    reviewer_list = []
    if reviewers:
        for reviewer in reviewers:
            new_rev = {}
            new_rev["email"] = reviewer["email"]
            new_rev["full_name"] = reviewer["full_name"]
            reviewer_list.append(new_rev)
    annot["reviewers"] = reviewer_list

    if text:
        annot["text"] = {"text": text}

    return annot


# List Darwin items in order to obtain item ids for external storage
def _list_items(api_key, dataset_id, team_slug, base_url):
    """
    List items in Darwin dataset, handling pagination.
    """
    url = f"{base_url}/{team_slug}/items?dataset_ids={dataset_id}"
    headers = {"accept": "application/json", "Authorization": f"ApiKey {api_key}"}
    items = []

    while url:
        response = requests.get(url, headers=headers)
        if response.ok:
            data = json.loads(response.text)
            items.extend(data["items"])
            next_page = data.get("page", {}).get("next")
            if next_page:
                url = f"{base_url}/{team_slug}/items?dataset_ids={dataset_id}&page[from]={next_page}"
            else:
                url = None
        else:
            raise requests.exceptions.HTTPError(
                f"GET request failed with status code {response.status_code}, {response.text}."
            )

    return items


def _51_to_v7_bbox(bbox_list, frame_size):
    """
    Converts 51 bounding box coordinates to V7 bounding box coordinates
    """
    width, height = frame_size
    new_bbox = {}
    new_bbox["x"] = bbox_list[0] * width
    new_bbox["y"] = bbox_list[1] * height
    new_bbox["w"] = bbox_list[2] * width
    new_bbox["h"] = bbox_list[3] * height

    return new_bbox


def _51_to_v7_keypoint(keypoint, frame_size):
    """
    Converts 51 keypoint coordinates to V7 keypoint coordinates
    """
    width, height = frame_size
    new_key = {}
    new_key["x"] = keypoint[0] * width
    new_key["y"] = keypoint[1] * height
    return new_key


def _v7_to_51_bbox(label, bbox_dict, height, width, attributes=None):
    """
    Converts V7 bounding box coordinates to 51 Detection
    """
    x = bbox_dict["x"] / width
    y = bbox_dict["y"] / height
    w = bbox_dict["w"] / width
    h = bbox_dict["h"] / height
    return fol.Detection(label=label, bounding_box=[x, y, w, h], **attributes)


def _v7_to_51_classification(label, attributes=None):
    """
    Converts a V7 classification to a 51 classification
    """
    return fol.Classification(label=label, **attributes)


def _51_to_v7_polygon(points, frame_size):
    """
    Converts a 51 polygon to a V7 polygon
    """
    width, height = frame_size
    new_poly = {
        "paths": [
            [{"x": x * width, "y": y * height} for x, y in point_list]
            for point_list in points
        ]
    }
    return new_poly


def _51_to_v7_polyline(polyline, frame_size):
    """
    Converts a 51 polygon line to a V7 polygon

    Note: Only curretly supports a single open line per annotation
    """
    width, height = frame_size
    new_polyline = {"path": [{"x": x * width, "y": y * height} for x, y in polyline[0]]}
    return new_polyline


def _v7_to_51_polyline(label, polyline, height, width, attributes=None):
    """
    Converts a V7 polygon to a 51 polygon

    Note: Open polylines are not currently supported by this integration
    """
    line_list = [
        [
            (keypoint["x"] / width, keypoint["y"] / height)
            for keypoint in polyline["path"]
        ]
    ]

    return fol.Polyline(
        label=label,
        points=line_list,
        closed=False,
        filled=False,
        **attributes,
    )


def _v7_to_51_keypoint(label, keypoint, height, width, attributes=None):
    """
    Converts V7 keypoint coordinates to 51 keypoint coordinates
    """
    new_keypoint = (keypoint["x"] / width, keypoint["y"] / height)
    return fol.Keypoint(label=label, points=[new_keypoint], **attributes)


def _v7_to_51_polygon(label, polygon, height, width, attributes=None):
    """
    Converts a V7 polygon to a 51 polygon
    """

    filled = True if len(polygon["paths"]) == 1 else False
    poly_list = [
        [(keypoint["x"] / width, keypoint["y"] / height) for keypoint in poly_path]
        for poly_path in polygon["paths"]
    ]

    return fol.Polyline(
        label=label,
        points=poly_list,
        closed=True,
        filled=filled,
        **attributes,
    )


def _chunk_list(lst, chunk_size):
    """Helper function to chunk a list into smaller lists of a specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def wait_until_items_finished_processing(dataset_id, team_slug, api_key, base_url):
    """
    Waits until all items in a dataset have finished processing before attempting to upload annotations
    """
    sleep_duration = 10
    while True:
        items = _list_items(api_key, dataset_id, team_slug, base_url)
        if not items:
            return
        if all(item["processing_status"] != "processing" for item in items):
            break
        logging.info(
            f"Waiting {sleep_duration} second for items to finish processing..."
        )
        time.sleep(sleep_duration)


_UNIQUE_TYPE_MAP = {
    "classification": "classifications",
    "classifications": "classifications",
    "instance": "segmentation",
    "instances": "segmentation",
    "polygons": "polylines",
    "polygon": "polylines",
    "polyline": "polylines",
    "polylines": "polylines",
    "keypoint": "keypoints",
    "keypoints": "keypoints",
}
