"""
LBP + SVM writer-identification pipeline.

Extracts 256-bin LBP histograms from text-line crops and trains an SVM
multi-class classifier.  Preserves the original algorithm described in the
project README.
"""

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from pre_processing.pre_processor import PreProcessor
from segmentation.line_segmentor import LineSegmentor
from features.feature_extractor import FeatureExtractor
from utils.constants import CHECKPOINTS_DIR, RESULTS_DIR, RANDOM_SEED, SVM_C, SVM_GAMMA

# Optional imports — sklearn is required for this module
try:
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    import joblib
    _SKLEARN_OK = True
except ImportError as _e:
    _SKLEARN_OK = False
    warnings.warn(f"scikit-learn/joblib not found: {_e}. LBPSVMTrainer unavailable.")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_features_from_image(
    image_path: str,
) -> Optional[np.ndarray]:
    """
    Load one form image, pre-process it, segment into lines, and
    return the aggregated LBP feature vector.

    Returns None if the image is corrupt or yields no lines.
    """
    gray_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if gray_img is None:
        return None

    gray_img, bin_img = PreProcessor.process(gray_img)
    if gray_img is None:
        return None

    gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()
    if not gray_lines:
        return None

    # Accumulate LBP histograms across all text lines
    total_hist = np.zeros(256, dtype=np.float64)
    for g_line, b_line in zip(gray_lines, bin_lines):
        if g_line.size == 0:
            continue
        total_hist = FeatureExtractor.get_lbp_histogram(
            g_line, b_line, total_hist, acc=True
        )

    # Normalise by mean bin value
    mean_val = np.mean(total_hist)
    if mean_val > 0:
        total_hist = total_hist / mean_val

    return total_hist


def build_feature_matrix(
    image_paths: List[str],
    labels:      List[int],
    verbose:     bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract LBP features for a list of images.

    Returns
    -------
    X      : ndarray (N_ok, 256) – feature matrix
    y      : ndarray (N_ok,)     – labels
    failed : list[int]           – indices of images that failed processing
    """
    X_list, y_list, failed = [], [], []

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if verbose and i % 50 == 0:
            print(f"  [{i + 1}/{len(image_paths)}] Processing {os.path.basename(path)}")

        feat = extract_features_from_image(path)

        if feat is None:
            failed.append(i)
            if verbose:
                print(f"    [!] Skipped (failed): {os.path.basename(path)}")
        else:
            X_list.append(feat)
            y_list.append(label)

    X = np.array(X_list) if X_list else np.zeros((0, 256))
    y = np.array(y_list) if y_list else np.zeros(0, dtype=int)
    return X, y, failed


# ---------------------------------------------------------------------------
# LBPSVMTrainer
# ---------------------------------------------------------------------------


class LBPSVMTrainer:
    """
    LBP feature extraction + SVM multi-class classifier.

    Usage
    -----
    trainer = LBPSVMTrainer()
    trainer.fit(train_paths, train_labels)
    metrics = trainer.evaluate(test_paths, test_labels)
    trainer.save("checkpoints/lbp_svm.pkl")
    """

    def __init__(
        self,
        C:     float = SVM_C,
        gamma: str   = SVM_GAMMA,
        seed:  int   = RANDOM_SEED,
    ):
        if not _SKLEARN_OK:
            raise ImportError("scikit-learn is required for LBPSVMTrainer.")

        self.C     = C
        self.gamma = gamma
        self.seed  = seed
        self.clf   = SVC(C=C, gamma=gamma, probability=True, random_state=seed)
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(
        self,
        image_paths: List[str],
        labels:      List[int],
        verbose:     bool = True,
    ) -> "LBPSVMTrainer":
        """
        Extract LBP features and train the SVM.

        Returns ``self`` for chaining.
        """
        print("[LBP+SVM] Extracting training features …")
        t0 = time.time()

        X, y, failed = build_feature_matrix(image_paths, labels, verbose)

        if len(X) == 0:
            raise ValueError("No training features could be extracted.")

        print(
            f"[LBP+SVM] Extracted {len(X)} feature vectors "
            f"({len(failed)} failed) in {time.time() - t0:.1f}s"
        )

        print("[LBP+SVM] Training SVM …")
        t1 = time.time()
        self.clf.fit(X, y)
        self.train_time_ = time.time() - t1
        self._fitted     = True

        print(f"[LBP+SVM] SVM trained in {self.train_time_:.2f}s")
        return self

    # ------------------------------------------------------------------

    def predict(self, image_paths: List[str]) -> np.ndarray:
        """Predict writer labels for a list of image paths."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        preds = []
        for path in image_paths:
            feat = extract_features_from_image(path)
            if feat is None:
                preds.append(-1)   # unknown / failed
            else:
                preds.append(int(self.clf.predict([feat])[0]))
        return np.array(preds)

    # ------------------------------------------------------------------

    def evaluate(
        self,
        image_paths: List[str],
        labels:      List[int],
        verbose:     bool = True,
    ) -> dict:
        """
        Extract features and compute classification metrics.

        Returns a dict with keys:
            accuracy, macro_f1, report, confusion_matrix,
            inference_time_per_image, n_failed
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate().")

        print("[LBP+SVM] Extracting evaluation features …")
        t0 = time.time()
        X, y_true, failed = build_feature_matrix(image_paths, labels, verbose)
        feature_time = time.time() - t0

        if len(X) == 0:
            raise ValueError("No evaluation features could be extracted.")

        t1 = time.time()
        y_pred = self.clf.predict(X)
        inference_time = (time.time() - t1) / max(len(X), 1)

        acc     = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report  = classification_report(y_true, y_pred, zero_division=0)
        cm      = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy":               acc,
            "macro_f1":               macro_f1,
            "report":                 report,
            "confusion_matrix":       cm,
            "inference_time_per_image_s": inference_time,
            "feature_extraction_time_s":  feature_time,
            "n_evaluated":            len(X),
            "n_failed":               len(failed),
        }

        if verbose:
            print(f"\n[LBP+SVM] Accuracy : {acc:.4f}")
            print(f"[LBP+SVM] Macro F1 : {macro_f1:.4f}")
            print("[LBP+SVM] Classification report:")
            print(report)

        return metrics

    # ------------------------------------------------------------------

    def save(self, path: str = "") -> str:
        """Persist model to disk using joblib. Returns the saved path."""
        if not path:
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            path = os.path.join(CHECKPOINTS_DIR, "lbp_svm.pkl")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.clf, path)
        print(f"[LBP+SVM] Model saved -> {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "LBPSVMTrainer":
        """Load a previously saved SVM model."""
        trainer = cls.__new__(cls)
        trainer.clf      = joblib.load(path)
        trainer._fitted  = True
        trainer.C        = trainer.clf.C
        trainer.gamma    = trainer.clf.gamma
        trainer.seed     = RANDOM_SEED
        print(f"[LBP+SVM] Model loaded ← {path}")
        return trainer

    # ------------------------------------------------------------------

    @property
    def model_size_bytes(self) -> int:
        """Approximate serialised model size in bytes."""
        import pickle, io
        buf = io.BytesIO()
        pickle.dump(self.clf, buf)
        return buf.tell()
