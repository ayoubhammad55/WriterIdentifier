"""
Pre-processor for IAM handwriting form images.

Extracts the handwritten paragraph region from a full IAM form scan.
"""

import cv2 as cv
import numpy as np

from utils.constants import DEBUG_PARAGRAPH_SEGMENTATION

try:
    from utils.utils import display_image
except ImportError:
    display_image = None  # graceful fallback


class PreProcessor:
    # Minimum image dimensions considered valid for processing.
    MIN_HEIGHT = 50
    MIN_WIDTH  = 100

    @staticmethod
    def process(gray_img: np.ndarray) -> tuple:
        """
        Pre-processes an IAM form image and extracts the handwritten paragraph.

        Parameters
        ----------
        gray_img : np.ndarray
            Grayscale IAM form image (H×W, dtype uint8).

        Returns
        -------
        (gray_out, bin_out) : tuple[np.ndarray, np.ndarray]
            Pre-processed gray and binary images.
            Returns (None, None) if the image is invalid or too small.
        """
        # ── Guard: reject None or non-array inputs ──────────────────────────
        if gray_img is None:
            return None, None

        if not isinstance(gray_img, np.ndarray):
            return None, None

        if gray_img.ndim != 2:
            # Convert colour -> gray just in case
            gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)

        h, w = gray_img.shape
        if h < PreProcessor.MIN_HEIGHT or w < PreProcessor.MIN_WIDTH:
            return None, None

        # ── Noise reduction ──────────────────────────────────────────────────
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        # ── Initial crop (remove left/right margins of the IAM form) ────────
        l_pad, r_pad = 150, 50
        if w > l_pad + r_pad + PreProcessor.MIN_WIDTH:
            gray_img = gray_img[:, l_pad:-r_pad]
        # If the image is too narrow to crop, keep it as-is.

        # ── Binarization (Otsu) ─────────────────────────────────────────────
        _, bin_img = cv.threshold(
            gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        # ── Crop paragraph region ────────────────────────────────────────────
        gray_img, bin_img = PreProcessor._crop_paragraph(gray_img, bin_img)

        # ── Final validity check ─────────────────────────────────────────────
        if gray_img is None or gray_img.size == 0:
            return None, None
        if gray_img.shape[0] < PreProcessor.MIN_HEIGHT // 2:
            return None, None

        return gray_img, bin_img

    # -------------------------------------------------------------------------

    @staticmethod
    def _crop_paragraph(
        gray_img: np.ndarray, bin_img: np.ndarray
    ) -> tuple:
        """
        Detects and crops the handwritten paragraph region inside an IAM form.

        Returns (None, None) if cropping fails.
        """
        height, width = gray_img.shape

        # ── Find horizontal black separator lines ────────────────────────────
        try:
            contours, _ = cv.findContours(
                bin_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
            )
        except Exception:
            return gray_img, bin_img  # can't detect contours; skip crop

        threshold_width = 1000
        line_offset     = 10
        up, down = 0, height - 1

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w < threshold_width:
                continue
            if y < height // 2:
                up = max(up, y + h + line_offset)
            else:
                down = min(down, y - line_offset)

        # ── Histograms for white-space trimming ──────────────────────────────
        kernel     = np.ones((3, 3), np.uint8)
        eroded_img = cv.erode(bin_img, kernel, iterations=2)

        hor_hist = np.sum(eroded_img, axis=1) / 255
        ver_hist = np.sum(eroded_img, axis=0) / 255

        left, right = 0, width - 1

        while left  < right  and ver_hist[left]  == 0: left  += 1
        while right > left   and ver_hist[right] == 0: right -= 1
        while up    < down   and hor_hist[up]    == 0: up    += 1
        while down  > up     and hor_hist[down]  == 0: down  -= 1

        # ── Safety: ensure non-empty crop ────────────────────────────────────
        if up >= down or left >= right:
            return gray_img, bin_img

        # ── Debug visualisation ──────────────────────────────────────────────
        if DEBUG_PARAGRAPH_SEGMENTATION and display_image is not None:
            img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
            cv.rectangle(img, (left, up), (right, down), (0, 0, 255), 3)
            display_image("Handwritten Paragraph", img)

        # ── Crop ─────────────────────────────────────────────────────────────
        gray_out = gray_img[up : down + 1, left : right + 1]
        bin_out  = bin_img [up : down + 1, left : right + 1]

        return gray_out, bin_out
