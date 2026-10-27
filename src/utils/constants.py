"""
Constants and path configuration for WriterIdentifier.

Kaggle-aware: automatically detects whether the project is running on Kaggle
and adapts input/output paths accordingly.
"""

import os

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_project_root() -> str:
    """Walk upward from this file until we find the project root (contains src/)."""
    current = os.path.dirname(os.path.abspath(__file__))  # src/utils/
    for _ in range(5):
        current = os.path.dirname(current)
        if os.path.isdir(os.path.join(current, "src")):
            return current
    # Fallback: two levels up from src/utils
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = _find_project_root()

# ---------------------------------------------------------------------------
# Kaggle vs. local paths
# ---------------------------------------------------------------------------

ON_KAGGLE = os.path.exists("/kaggle")

if ON_KAGGLE:
    # -----------------------------------------------------------------------
    # Kaggle: input data is read-only; write outputs to /kaggle/working/
    # -----------------------------------------------------------------------
    KAGGLE_INPUT  = "/kaggle/input"
    KAGGLE_WORKING = "/kaggle/working"

    # The user must attach the IAM dataset on Kaggle.
    # Dataset slug should be named "iam-handwriting-database" (adjust if needed).
    # We try common dataset slugs; the notebook will validate the path.
    _candidate_dirs = [
        "/kaggle/input/iam-handwriting-database",
        "/kaggle/input/iam-handwriting",
        "/kaggle/input/iam-dataset",
        "/kaggle/input/iam",
    ]
    IAM_DATASET_ROOT = next(
        (d for d in _candidate_dirs if os.path.isdir(d)),
        "/kaggle/input/iam-handwriting-database",  # default (may not exist yet)
    )

    IAM_FORMS_PATH = os.path.join(IAM_DATASET_ROOT, "forms")
    IAM_XML_PATH   = os.path.join(IAM_DATASET_ROOT, "xml")

    OUTPUT_DIR       = KAGGLE_WORKING
    CHECKPOINTS_DIR  = os.path.join(KAGGLE_WORKING, "checkpoints")
    RESULTS_DIR      = os.path.join(KAGGLE_WORKING, "results")
    DATASET_PATH     = os.path.join(KAGGLE_WORKING, "dataset") + "/"
    TESTCASES_PATH   = os.path.join(KAGGLE_WORKING, "testcases") + "/"

else:
    # -----------------------------------------------------------------------
    # Local development paths
    # -----------------------------------------------------------------------
    IAM_DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
    IAM_FORMS_PATH   = os.path.join(IAM_DATASET_ROOT, "form")
    IAM_XML_PATH     = os.path.join(IAM_DATASET_ROOT, "xml")

    OUTPUT_DIR       = os.path.join(PROJECT_ROOT, "outputs")
    CHECKPOINTS_DIR  = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_DIR      = os.path.join(PROJECT_ROOT, "results")
    DATASET_PATH     = os.path.join(PROJECT_ROOT, "data", "dataset") + "/"
    TESTCASES_PATH   = os.path.join(PROJECT_ROOT, "data", "testcases") + "/"

# ---------------------------------------------------------------------------
# Result files (original API, now using absolute paths)
# ---------------------------------------------------------------------------

EXPECTED_RESULTS_FILENAME = "results_expected.txt"
EXPECTED_RESULTS_PATH     = os.path.join(TESTCASES_PATH, EXPECTED_RESULTS_FILENAME)
PREDICTED_RESULTS_PATH    = os.path.join(TESTCASES_PATH, "results_predicted.txt")
WRONG_CASES_PATH          = os.path.join(OUTPUT_DIR, "wrong") + "/"

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

# Set True to regenerate test iterations from the IAM dataset.
GENERATE_TEST_ITERATIONS = False

# Set True once to regenerate pre-processed images (saves future time).
GENERATE_PRE_PROCESSED_DATA = False

# ---------------------------------------------------------------------------
# Debugging flags
# ---------------------------------------------------------------------------

DEBUG_PARAGRAPH_SEGMENTATION = False
DEBUG_LINE_SEGMENTATION      = False
DEBUG_THINNING_ALGORITHM     = False
DEBUG_SAVE_WRONG_TESTCASES   = False

# ---------------------------------------------------------------------------
# Training / model defaults
# ---------------------------------------------------------------------------

RANDOM_SEED  = 42
IMAGE_HEIGHT = 128   # pixels – resize target for CNN / ResNet inputs
IMAGE_WIDTH  = 512   # pixels

# SVM
SVM_C     = 5.0
SVM_GAMMA = "auto"

# CNN / ResNet
BATCH_SIZE     = 16
NUM_EPOCHS     = 30
LEARNING_RATE  = 1e-3
VAL_SPLIT      = 0.15   # fraction of training data used for validation
TEST_SPLIT     = 0.15   # fraction of the full dataset held out as test set
