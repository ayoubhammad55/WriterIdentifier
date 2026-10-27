"""
Original LBP+SVM writer identification pipeline.

This file preserves the original prediction / accuracy-checking workflow.
For full Kaggle training experiments, use the notebooks/ directory.

Run from the project root:
    python src/main.py [--testcases path/to/testcases]
"""

import argparse
import os
import sys
import time

import numpy as np

# Ensure the src/ directory is on the path when running as a script
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sklearn.svm import SVC

from data.test_generator import TestGenerator
from pre_processing.pre_processor import PreProcessor
from segmentation.line_segmentor import LineSegmentor
from features.feature_extractor import FeatureExtractor
from utils.utils import calculate_accuracy, print_wrong_testcases, save_wrong_testcases
from utils.constants import (
    GENERATE_TEST_ITERATIONS,
    DEBUG_SAVE_WRONG_TESTCASES,
    TESTCASES_PATH,
    PREDICTED_RESULTS_PATH,
)

import cv2 as cv


# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────

result_file   = None
total_time    = 0.0
testcase_time = []


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline functions
# ─────────────────────────────────────────────────────────────────────────────


def run(testcases_path: str = TESTCASES_PATH):
    global result_file, total_time, testcase_time

    # Create output directory if necessary
    os.makedirs(os.path.dirname(PREDICTED_RESULTS_PATH), exist_ok=True)

    start       = time.time()
    result_file = open(PREDICTED_RESULTS_PATH, "w")

    for root, dirs, files in os.walk(testcases_path):
        for d in sorted(dirs):
            print("Running test iteration '%s'..." % d)
            t = time.time()
            process_testcase(os.path.join(testcases_path, d) + os.sep)
            t = time.time() - t
            testcase_time.append(t)
            print("Finish test iteration '%s' in %.02f seconds\n" % (d, t))
        break

    result_file.close()
    total_time = time.time() - start


def process_testcase(path: str):
    features, labels = [], []

    for root, dirs, files in os.walk(path):
        for d in sorted(dirs):
            print("    Processing writer '%s'..." % d)
            x, y = get_writer_features(os.path.join(path, d) + os.sep, d)
            features.extend(x)
            labels.extend(y)
        break

    if not features:
        print("    [!] No features extracted for this testcase — skipping.")
        result_file.write("unknown\n")
        return

    classifier = SVC(C=5.0, gamma="auto", probability=True)
    classifier.fit(features, labels)

    for root, dirs, files in os.walk(path):
        for filename in files:
            x = get_features(os.path.join(path, filename))

            if not x:
                result_file.write("unknown\n")
                print("    [!] Could not extract features for '%s'" % filename)
                continue

            p = classifier.predict_proba(x)
            p = np.sum(p, axis=0)
            r = classifier.classes_[np.argmax(p)]

            result_file.write(str(r) + "\n")
            print("    Classifying '%s' as writer '%s'" % (filename, r))
        break


def get_writer_features(path: str, writer_id: str):
    total_gray_lines, total_bin_lines = [], []

    for root, dirs, files in os.walk(path):
        for filename in files:
            fp = os.path.join(path, filename)
            gray_img = cv.imread(fp, cv.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue
            gray_img, bin_img = PreProcessor.process(gray_img)
            if gray_img is None:
                continue
            gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()
            total_gray_lines.extend(gray_lines)
            total_bin_lines.extend(bin_lines)
        break

    x, y = [], []
    for g, b in zip(total_gray_lines, total_bin_lines):
        f = FeatureExtractor([g], [b]).extract()
        x.append(f)
        y.append(writer_id)

    return x, y


def get_features(path: str):
    gray_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if gray_img is None:
        return []

    gray_img, bin_img = PreProcessor.process(gray_img)
    if gray_img is None:
        return []

    gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()

    x = []
    for g, b in zip(gray_lines, bin_lines):
        f = FeatureExtractor([g], [b]).extract()
        x.append(f)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Writer Identifier — LBP+SVM baseline")
    parser.add_argument(
        "--testcases",
        default=TESTCASES_PATH,
        help="Path to the testcases directory (default: %(default)s)",
    )
    args = parser.parse_args()

    if GENERATE_TEST_ITERATIONS:
        gen = TestGenerator()
        gen.generate(args.testcases, 10, 3, 2)

    run(args.testcases)

    print("-------------------------------")
    print("Total elapsed time: %.2f seconds" % total_time)
    if testcase_time:
        print("Average testcase time: %.2f seconds" % np.average(testcase_time))
    acc = calculate_accuracy()
    if isinstance(acc, tuple) and acc[0] != -1:
        print("Classification accuracy: %d/%d" % acc)
    print("-------------------------------")

    if DEBUG_SAVE_WRONG_TESTCASES:
        print_wrong_testcases()
        save_wrong_testcases()
