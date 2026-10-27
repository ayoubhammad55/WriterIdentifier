"""
IAM Handwriting Dataset Loader.

Supports the standard IAM offline handwriting database layout:

    <root>/
        forms/          (or form/)   – PNG/TIFF form scans
        xml/                         – XML annotation files (one per form)

Writer IDs are parsed from the XML files (attribute ``writer-id``).

Kaggle note
-----------
Attach the IAM dataset under a dataset named "iam-handwriting-database"
(or another slug) and set ``IAM_FORMS_PATH`` / ``IAM_XML_PATH`` in
``utils/constants.py`` accordingly, or pass explicit paths to the loader.
"""

import os
import xml.etree.ElementTree as ET
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import cv2 as cv
import numpy as np

from utils.constants import (
    IAM_FORMS_PATH,
    IAM_XML_PATH,
    RANDOM_SEED,
    TEST_SPLIT,
    VAL_SPLIT,
)
from pre_processing.pre_processor import PreProcessor


# ---------------------------------------------------------------------------
# Known problematic form images (corrupt / missing scan regions)
# ---------------------------------------------------------------------------
_SKIP_FORMS = {
    "h07-084",
    "h07-080a",
    "h07-075a",
    "h07-078a",
    "k03-164",
    "k02-053",
    "j07-005",
}

# Accepted image extensions
_FORM_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_dataset_paths(
    forms_path: str = IAM_FORMS_PATH,
    xml_path:   str = IAM_XML_PATH,
) -> bool:
    """
    Check whether the IAM dataset directories exist and are non-empty.

    Prints a clear diagnostic message when files are missing.

    Returns True if the dataset looks usable.
    """
    ok = True

    if not os.path.isdir(forms_path):
        print(
            "[IAM] [MISSING] Forms directory not found: " + forms_path + "\n"
            "      -> Attach the IAM dataset on Kaggle or set IAM_FORMS_PATH."
        )
        ok = False
    else:
        n_forms = sum(
            1
            for root, _, files in os.walk(forms_path)
            for f in files
            if os.path.splitext(f)[1].lower() in _FORM_EXTENSIONS
        )
        print("[IAM] [OK] Forms directory found (" + str(n_forms) + " images): " + forms_path)
        if n_forms == 0:
            ok = False

    if not os.path.isdir(xml_path):
        print(
            "[IAM] [MISSING] XML directory not found: " + xml_path + "\n"
            "      -> Attach the IAM dataset on Kaggle or set IAM_XML_PATH."
        )
        ok = False
    else:
        n_xml = sum(
            1 for root, _, files in os.walk(xml_path)
            for f in files if f.lower().endswith(".xml")
        )
        print("[IAM] [OK] XML directory found (" + str(n_xml) + " files): " + xml_path)
        if n_xml == 0:
            ok = False

    return ok


def get_writer_id_from_xml(form_id: str, xml_path: str = IAM_XML_PATH) -> Optional[int]:
    """
    Parse the writer-id attribute from the XML annotation file for ``form_id``.

    Parameters
    ----------
    form_id  : str  – form stem without extension, e.g. "a01-000u"
    xml_path : str  – directory containing the XML files

    Returns
    -------
    writer_id : int | None – None if the XML file is missing or malformed
    """
    xml_file = os.path.join(xml_path, form_id + ".xml")
    if not os.path.isfile(xml_file):
        # Some IAM mirrors keep XML annotations in nested directories.
        xml_file = None
        for directory, _, filenames in os.walk(xml_path):
            candidate = form_id + ".xml"
            if candidate in filenames:
                xml_file = os.path.join(directory, candidate)
                break
        if xml_file is None:
            return None
    try:
        root = ET.parse(xml_file).getroot()
        value = root.attrib.get("writer-id")
        return int(value) if value is not None else None
    except (OSError, ET.ParseError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------


class IAMLoader:
    """
    Loads IAM form images, resolves writer labels from XML, and produces
    train / validation / test splits with no document-level leakage.
    """

    def __init__(
        self,
        forms_path: str = IAM_FORMS_PATH,
        xml_path:   str = IAM_XML_PATH,
        skip_forms: Optional[set] = None,
        min_forms_per_writer: int = 2,
        seed: int = RANDOM_SEED,
    ):
        self.forms_path  = forms_path
        self.xml_path    = xml_path
        self.skip_forms  = skip_forms if skip_forms is not None else _SKIP_FORMS
        self.min_forms   = min_forms_per_writer
        self.seed        = seed

        # Populated by load()
        self.form_paths:   List[str] = []      # absolute paths
        self.form_ids:     List[str] = []      # stem without extension
        self.writer_ids:   List[int] = []      # parallel to form_paths

        # Splits (indices into form_paths)
        self.train_idx: List[int] = []
        self.val_idx:   List[int] = []
        self.test_idx:  List[int] = []

        # Label encoding
        self.writer_to_label: Dict[int, int] = {}
        self.label_to_writer: Dict[int, int] = {}
        self.num_classes: int = 0

    # ------------------------------------------------------------------

    def load(self) -> "IAMLoader":
        """
        Discover all form images, resolve writer IDs, filter rare writers,
        and build reproducible train/val/test splits.

        Returns ``self`` for chaining.
        """
        if not os.path.isdir(self.forms_path):
            raise FileNotFoundError(
                f"Forms directory not found: {self.forms_path}\n"
                "Attach the IAM dataset and check IAM_FORMS_PATH in constants.py."
            )

        # ── 1. Discover images & resolve labels ──────────────────────────
        writer_to_forms: Dict[int, List[str]] = defaultdict(list)

        # Index XML files once so nested dataset layouts work without repeatedly
        # scanning the annotation tree for every form image.
        xml_by_stem = {}
        for xml_root, _, xml_files in os.walk(self.xml_path):
            for xml_name in xml_files:
                if xml_name.lower().endswith(".xml"):
                    xml_by_stem.setdefault(os.path.splitext(xml_name)[0],
                                           os.path.join(xml_root, xml_name))

        image_files = []
        for image_root, _, image_names in os.walk(self.forms_path):
            for fname in image_names:
                stem, ext = os.path.splitext(fname)
                if ext.lower() in _FORM_EXTENSIONS:
                    image_files.append((stem, os.path.join(image_root, fname)))

        for stem, abs_path in sorted(image_files):
            if stem in self.skip_forms:
                continue

            xml_file = xml_by_stem.get(stem)
            writer_id = None
            if xml_file:
                try:
                    root = ET.parse(xml_file).getroot()
                    raw_writer_id = root.attrib.get("writer-id")
                    writer_id = int(raw_writer_id) if raw_writer_id is not None else None
                except (OSError, ET.ParseError, ValueError):
                    writer_id = None

            if writer_id is None:
                # Try alternate: some datasets use the xml dir differently
                print(f"[IAM] [!] No XML for {fname} — skipping.")
                continue

            writer_to_forms[writer_id].append(abs_path)

        # ── 2. Filter writers with too few forms ─────────────────────────
        filtered: Dict[int, List[str]] = {
            wid: paths
            for wid, paths in writer_to_forms.items()
            if len(paths) >= self.min_forms
        }
        print(
            f"[IAM] Writers after filtering (≥{self.min_forms} forms): "
            f"{len(filtered)} / {len(writer_to_forms)}"
        )

        # ── 3. Encode labels ──────────────────────────────────────────────
        sorted_writers = sorted(filtered.keys())
        self.writer_to_label = {w: i for i, w in enumerate(sorted_writers)}
        self.label_to_writer = {i: w for w, i in self.writer_to_label.items()}
        self.num_classes      = len(sorted_writers)

        # ── 4. Flatten lists ──────────────────────────────────────────────
        all_paths, all_labels = [], []
        for wid in sorted_writers:
            for p in filtered[wid]:
                all_paths.append(p)
                all_labels.append(self.writer_to_label[wid])

        self.form_paths = all_paths
        self.form_ids   = [os.path.splitext(os.path.basename(p))[0] for p in all_paths]
        self.writer_ids = all_labels

        # ── 5. Reproducible splits (stratified by writer, no leakage) ────
        rng = random.Random(self.seed)
        writer_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(all_labels):
            writer_indices[label].append(idx)

        train_idx, val_idx, test_idx = [], [], []
        for label, indices in writer_indices.items():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = max(1, round(n * TEST_SPLIT))
            n_val  = max(1, round(n * VAL_SPLIT))
            n_train = n - n_test - n_val
            if n_train < 1:
                # Too few samples — put everything in train
                train_idx.extend(shuffled)
                continue
            train_idx.extend(shuffled[:n_train])
            val_idx  .extend(shuffled[n_train : n_train + n_val])
            test_idx .extend(shuffled[n_train + n_val :])

        self.train_idx = train_idx
        self.val_idx   = val_idx
        self.test_idx  = test_idx

        print(
            f"[IAM] Dataset split — "
            f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)} "
            f"| {self.num_classes} writers"
        )
        return self

    # ------------------------------------------------------------------

    def get_split(self, split: str) -> Tuple[List[str], List[int]]:
        """
        Return (image_paths, labels) for a given split name
        ('train', 'val', or 'test').
        """
        mapping = {
            "train": self.train_idx,
            "val":   self.val_idx,
            "test":  self.test_idx,
        }
        if split not in mapping:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
        idx = mapping[split]
        return (
            [self.form_paths[i] for i in idx],
            [self.writer_ids[i] for i in idx],
        )

    # ------------------------------------------------------------------
    # Legacy helper: pre-process and save processed images
    # ------------------------------------------------------------------

    @staticmethod
    def generate_processed_data(
        forms_path:          str = IAM_FORMS_PATH,
        xml_path:            str = IAM_XML_PATH,
        processed_gray_path: str = "",
        processed_bin_path:  str = "",
        labels_file:         str = "",
    ) -> dict:
        """
        Pre-processes all IAM form images and writes results to disk.

        Returns a dict mapping filename -> writer_id.
        """
        labels: dict = {}

        os.makedirs(processed_gray_path, exist_ok=True)
        os.makedirs(processed_bin_path,  exist_ok=True)

        image_files = []
        for image_root, _, image_names in os.walk(forms_path):
            for fname in image_names:
                stem, ext = os.path.splitext(fname)
                if ext.lower() in _FORM_EXTENSIONS:
                    image_files.append((stem, os.path.join(image_root, fname)))

        for stem, abs_path in sorted(image_files):
            fname = os.path.basename(abs_path)
            if stem in _SKIP_FORMS:
                continue

            writer_id = get_writer_id_from_xml(stem, xml_path)
            if writer_id is None:
                print(f"  Skipping {fname} (no XML).")
                continue

            labels[fname] = writer_id
            print(f"  Loading {fname} (writer {writer_id}) …")

            img = cv.imread(abs_path, cv.IMREAD_GRAYSCALE)
            gray_out, bin_out = PreProcessor.process(img)

            if gray_out is None:
                print(f"  [!] Pre-processing failed for {fname}.")
                continue

            if processed_gray_path:
                cv.imwrite(os.path.join(processed_gray_path, fname), gray_out)
            if processed_bin_path:
                cv.imwrite(os.path.join(processed_bin_path,  fname), bin_out)

        if labels_file:
            with open(labels_file, "w") as f:
                f.write(str(labels))

        return labels
