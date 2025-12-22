#!/usr/bin/env python3
"""
Standalone "notebook flow" runner:

1) Load/create a FiftyOne dataset (optionally from the Zoo)
2) Optionally launch the FiftyOne App
3) Start a Darwin annotation run via `dataset.annotate(..., backend="darwin")`
   and optionally open the Darwin editor
4) Later, pull the Darwin annotations back into FiftyOne via `dataset.load_annotations(anno_key)`

This script reads Darwin credentials from:
  - env: DARWIN_API_KEY or V7_API_KEY
  - otherwise: ~/.fiftyone/annotation_config.json -> backends.darwin.api_key/base_url

Examples:
  # Start a run (create local dataset from zoo, open Darwin editor)
  python scripts/voxel_51_v7_flow.py start \
    --zoo quickstart \
    --dataset-name demonstration-dataset \
    --dataset-slug v51 \
    --label-field darwin_detections \
    --classes apple,orange \
    --launch-editor

  # Pull results back into FiftyOne later
  python scripts/voxel_51_v7_flow.py pull \
    --dataset-name demonstration-dataset \
    --anno-key <PASTE_ANNO_KEY> \
    --label-field darwin_detections
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from pprint import pprint
import re
from urllib.parse import urlparse, urlunparse

import faulthandler

faulthandler.enable()

# Make repo importable when run via `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fiftyone as fo
import fiftyone.zoo as foz


DEFAULT_TEAM_API_URL = "http://localhost:4000/api/v2/teams"


def load_darwin_backend_config() -> dict:
    path = os.path.expanduser("~/.fiftyone/annotation_config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        backends = data.get("backends") or {}
        cfg = backends.get("darwin") or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def derive_darwin_host_from_team_api_url(team_api_url: str) -> str:
    marker = "/api/v2/teams"
    if not team_api_url:
        return ""
    if marker in team_api_url:
        return team_api_url.split(marker, 1)[0].rstrip("/")
    return team_api_url.rstrip("/")


def ensure_darwin_py_points_at_correct_host(team_api_url: str) -> None:
    """
    darwin-py validates the API key against `${DARWIN_BASE_URL}/api/users/token_info`.
    If you're using a local Darwin instance, set DARWIN_BASE_URL accordingly.
    """
    derived = derive_darwin_host_from_team_api_url(team_api_url)
    if not derived:
        return

    current = os.environ.get("DARWIN_BASE_URL")
    # Important: users often switch between localhost/staging; if DARWIN_BASE_URL is
    # already set, it can silently point darwin-py at the wrong host. Prefer
    # consistency with the configured team API URL.
    if current and current.rstrip("/") != derived.rstrip("/"):
        # Avoid printing full URLs in non-verbose mode; caller will handle verbosity
        ...
    os.environ["DARWIN_BASE_URL"] = derived


def get_api_key_and_base_url(args: argparse.Namespace) -> tuple[str, str]:
    cfg = load_darwin_backend_config()
    api_key = args.api_key or os.environ.get("DARWIN_API_KEY") or os.environ.get("V7_API_KEY") or cfg.get("api_key")
    base_url = args.base_url or cfg.get("base_url") or DEFAULT_TEAM_API_URL
    if not api_key:
        raise RuntimeError(
            "No Darwin API key found. Set env DARWIN_API_KEY (or V7_API_KEY), or add "
            "backends.darwin.api_key in ~/.fiftyone/annotation_config.json"
        )
    return api_key, base_url


def _safe_url(url: str) -> str:
    """Returns scheme://netloc for a URL (no path/query), or the original if parsing fails."""
    try:
        p = urlparse(url)
        if p.scheme and p.netloc:
            return urlunparse((p.scheme, p.netloc, "", "", "", ""))
    except Exception:
        ...
    return url


def _redact_text(text: str) -> str:
    """
    Redact common sensitive data from logs:
    - presigned URL querystrings (e.g. X-Amz-Signature=...)
    - raw ApiKey headers in error messages
    """
    if not text:
        return text

    # Remove common presigned URL params
    text = re.sub(r"(X-Amz-Signature=)[^&\\s]+", r"\\1<redacted>", text)
    text = re.sub(r"(X-Amz-Credential=)[^&\\s]+", r"\\1<redacted>", text)
    text = re.sub(r"(X-Amz-Security-Token=)[^&\\s]+", r"\\1<redacted>", text)

    # Redact ApiKey tokens
    text = re.sub(r"(Authorization:\\s*ApiKey)\\s+[^\\s]+", r"\\1 <redacted>", text, flags=re.IGNORECASE)
    text = re.sub(r"(ApiKey)\\s+[^\\s]+", r"\\1 <redacted>", text, flags=re.IGNORECASE)
    return text


def load_or_create_dataset(dataset_name: str, zoo: str | None, max_samples: int) -> fo.Dataset:
    if dataset_name in fo.list_datasets():
        ds = fo.load_dataset(dataset_name)
        # Prevent FiftyOne's atexit cleanup from deleting this dataset between commands
        ds.persistent = True
        return ds
    if not zoo:
        raise RuntimeError(
            f"Local FiftyOne dataset '{dataset_name}' does not exist. Provide --zoo to create it."
        )
    ds = foz.load_zoo_dataset(zoo, max_samples=max_samples, dataset_name=dataset_name)
    # Prevent FiftyOne's atexit cleanup from deleting this dataset between commands
    ds.persistent = True
    return ds


def build_label_schema(label_field: str, label_type: str, classes_csv: str) -> dict:
    classes = sorted({c.strip() for c in classes_csv.split(",") if c.strip()})
    if not classes:
        raise RuntimeError("--classes is required when starting a new annotation field (e.g. darwin_detections)")
    return {label_field: {"type": label_type, "classes": classes}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-name", required=True, help="Local FiftyOne dataset name")
    common.add_argument("--label-field", default="darwin_detections", help="FiftyOne field to write Darwin labels into")
    common.add_argument("--label-type", default="detections", help="Label type: detections, polylines, polygons, ...")
    common.add_argument("--dataset-slug", required=True, help="Darwin dataset slug (remote)")
    common.add_argument("--api-key", default="", help="Override Darwin API key (otherwise uses annotation_config.json)")
    common.add_argument("--base-url", default="", help="Override Darwin team API url (e.g. http://localhost:4000/api/v2/teams)")
    common.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output (may include sensitive URLs/paths).",
    )

    start = sub.add_parser("start", parents=[common], help="Start a Darwin annotation run and optionally open the editor")
    start.add_argument("--zoo", default="", help="Optional zoo dataset name used to create the local dataset if missing")
    start.add_argument("--max-samples", type=int, default=10)
    start.add_argument("--view-samples", type=int, default=5)
    start.add_argument("--no-app", action="store_true", help="Do not launch the FiftyOne App")
    start.add_argument("--keep-alive", type=int, default=0, help="Keep process alive N seconds after launching app")
    start.add_argument("--classes", default="", help="Comma-separated class list (required for start)")
    start.add_argument("--launch-editor", action="store_true", help="Open Darwin editor in browser after starting run")

    pull = sub.add_parser("pull", parents=[common], help="Pull Darwin annotations back into FiftyOne")
    pull.add_argument("--anno-key", required=True, help="FiftyOne annotation run key (printed by `start`)")
    pull.add_argument("--reset-label-field", action="store_true", help="Delete label field before pulling (avoid duplicates)")

    status = sub.add_parser("status", parents=[common], help="Check Darwin status counts for a run")
    status.add_argument("--anno-key", required=True, help="FiftyOne annotation run key (printed by `start`)")

    verify = sub.add_parser(
        "verify",
        parents=[common],
        help="Verify whether the local FiftyOne label field contains duplicate detections",
    )
    verify.add_argument(
        "--max-show",
        type=int,
        default=10,
        help="Maximum number of samples with duplicates to print",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key, base_url = get_api_key_and_base_url(args)
    prev_darwin_base = os.environ.get("DARWIN_BASE_URL")
    ensure_darwin_py_points_at_correct_host(base_url)
    if args.verbose and prev_darwin_base and prev_darwin_base.rstrip("/") != os.environ.get("DARWIN_BASE_URL", "").rstrip("/"):
        print(f"Overriding DARWIN_BASE_URL: {prev_darwin_base} -> {os.environ.get('DARWIN_BASE_URL')}")

    print("Python:", sys.version)
    if args.verbose:
        print("Python executable:", sys.executable)
    print("FiftyOne:", getattr(fo, "__version__", "<unknown>"))
    print("dataset_name:", args.dataset_name)
    print("darwin dataset_slug:", args.dataset_slug)
    # Avoid printing full URLs/env by default
    print("darwin host:", _safe_url(base_url))
    if args.verbose:
        print("darwin base_url (full):", base_url)
        print("DARWIN_BASE_URL env:", os.environ.get("DARWIN_BASE_URL"))
        print("FIFTYONE_CONFIG_PATH env:", os.environ.get("FIFTYONE_CONFIG_PATH"))
        print("FIFTYONE_DATABASE_URI env:", os.environ.get("FIFTYONE_DATABASE_URI"))
        try:
            print("FiftyOne config.database_uri:", fo.config.database_uri)
        except Exception:
            ...

    dataset = load_or_create_dataset(
        args.dataset_name,
        zoo=getattr(args, "zoo", "") or None,
        max_samples=getattr(args, "max_samples", 10),
    )
    print("\n--- dataset ---")
    if args.verbose:
        print(dataset)
    else:
        print(f"Name: {dataset.name} | media_type: {dataset.media_type} | samples: {len(dataset)} | persistent: {getattr(dataset, 'persistent', None)}")
    try:
        print("persistent:", dataset.persistent)
    except Exception:
        pass

    if args.cmd == "start":
        view = dataset.take(args.view_samples)
        print("\n--- view ---")
        if args.verbose:
            print(view)
        else:
            try:
                print(f"View samples: {view.count()}")
            except Exception:
                print("View created")

        session = None
        if not args.no_app:
            session = fo.launch_app(view)
            print("FiftyOne App:", session)

        label_schema = build_label_schema(args.label_field, args.label_type, args.classes)
        print("\n--- label_schema ---")
        # Safe to print ontology/classes; avoid printing large debug structures
        print({args.label_field: {"type": args.label_type, "classes": label_schema[args.label_field]["classes"]}})

        from uuid import uuid4

        anno_key = f"key_{str(uuid4()).replace('-', '_')}"

        print("\n=== Starting Darwin annotation run ===")
        print("anno_key:", anno_key)
        try:
            results = dataset.annotate(
                anno_key,
                label_schema=label_schema,
                backend="darwin",
                api_key=api_key,
                base_url=base_url,
                dataset_slug=args.dataset_slug,
                launch_editor=args.launch_editor,
            )
            print("annotate() OK")
            # Avoid printing item maps / ids by default
            if args.verbose:
                print("results:", results)
        except Exception:
            print("annotate() FAILED")
            if args.verbose:
                traceback.print_exc()
            else:
                print(_redact_text(traceback.format_exc()))
            return 2

        print("\nNext:")
        print(f"- annotate in Darwin, then pull back via:\n  python scripts/voxel_51_v7_flow.py pull --dataset-name {args.dataset_name} --dataset-slug {args.dataset_slug} --anno-key {anno_key} --label-field {args.label_field}")

        if args.keep_alive and args.keep_alive > 0:
            print(f"\nKeeping process alive for {args.keep_alive}s so you can inspect (Ctrl+C to stop).")
            if session is not None:
                try:
                    print("FiftyOne Session URL:", session.url)
                except Exception:
                    pass
            try:
                time.sleep(args.keep_alive)
            except KeyboardInterrupt:
                print("\nInterrupted; exiting.")
        return 0

    if args.cmd == "status":
        results = dataset.load_annotation_results(args.anno_key)
        results.check_status()
        return 0

    if args.cmd == "pull":
        if args.reset_label_field:
            if args.label_field in dataset.get_field_schema():
                dataset.delete_sample_field(args.label_field)
                print(f"Deleted local field `{args.label_field}` before pull")
        try:
            dataset.load_annotations(args.anno_key)
            print("load_annotations() OK")
        except Exception as e:
            # Common local-Darwin failure: export ZIP is hosted on localstack/minio
            # with virtual-hosted-style bucket domains like:
            #   http://<bucket>.s3.<region>.localhost:4566/...
            # which requires DNS/hosts to resolve that subdomain to 127.0.0.1.
            msg = _redact_text(str(e))
            if "Failed to resolve" in msg and ".localhost" in msg:
                # Try to extract the failing hostname from urllib3/requests errors
                host = None
                m = re.search(r"Failed to resolve '([^']+)'", msg)
                if m:
                    host = m.group(1)
                print("\nPull failed due to DNS resolution for local export host.")
                print("Your Darwin export URL points at a host like `<bucket>.s3.<region>.localhost:4566`.")
                if host:
                    print(f"\nHostname to resolve: {host}")
                    print("macOS quick fix (adds a hosts entry):")
                    print(f"  sudo sh -c \"echo '127.0.0.1 {host}' >> /etc/hosts\"")
                    print("  sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder")
                print("Fix options:")
                print("- Add an /etc/hosts entry mapping that exact host to 127.0.0.1")
                print("  (or configure DNS so `*.localhost` resolves to 127.0.0.1)")
                print("- Ensure localstack/minio is running and reachable on port 4566")
                print("\nOriginal error:")
            if args.verbose:
                traceback.print_exc()
            else:
                print(_redact_text(traceback.format_exc()))
            return 3
        return 0

    if args.cmd == "verify":
        field = args.label_field
        print(f"\n=== Verifying duplicates in field `{field}` ===")
        if field not in dataset.get_field_schema():
            print(f"Field `{field}` not found on dataset")
            return 4

        # For detections, count duplicates by (label, bbox) signature
        total = 0
        unique_total = 0
        dup_total = 0
        shown = 0

        for s in dataset.select_fields(["filepath", field]):
            try:
                val = s[field]
            except Exception:
                val = None
            dets = getattr(val, "detections", None) if val is not None else None
            if not dets:
                continue

            sigs = []
            for d in dets:
                bbox = getattr(d, "bounding_box", None)
                if bbox is None:
                    # fallback: label only
                    sig = (getattr(d, "label", None), None)
                else:
                    # round bbox for stability
                    sig = (
                        getattr(d, "label", None),
                        tuple(round(float(x), 6) for x in bbox),
                    )
                sigs.append(sig)

            total += len(sigs)
            uniq = set(sigs)
            unique_total += len(uniq)
            if len(uniq) != len(sigs):
                dup = len(sigs) - len(uniq)
                dup_total += dup
                if shown < args.max_show:
                    shown += 1
                    print(f"- {s.filepath}: detections={len(sigs)} unique={len(uniq)} dups={dup}")

        print("\nSummary:")
        print("total detections:", total)
        print("unique signatures:", unique_total)
        print("duplicate detections:", dup_total)
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())



