"""
LBP feature extractor for handwritten text lines.

Preserves the original implementation (vectorised 8-neighbour LBP with
distance v=3) while fixing the boolean->uint8 bitwise-OR issue and
updating the cv.calcHist mask handling.
"""

import cv2 as cv
import numpy as np


class FeatureExtractor:
    def __init__(self, gray_lines: list, bin_lines: list):
        """
        Parameters
        ----------
        gray_lines : list[ndarray]  – grayscale text-line crops
        bin_lines  : list[ndarray]  – binary (0/255) text-line crops
        """
        self.gray_lines = gray_lines
        self.bin_lines  = bin_lines
        self.features   = []

    # ------------------------------------------------------------------

    def extract(self) -> list:
        """Return the 256-bin LBP feature vector for the supplied lines."""
        self.features = []
        self.features.extend(self.lbp_histogram())
        return self.features

    # ------------------------------------------------------------------

    def lbp_histogram(self) -> np.ndarray:
        """
        Compute the accumulated, mean-normalised LBP histogram over all
        text lines.
        """
        hist = np.zeros(256, dtype=np.float64)

        for g, b in zip(self.gray_lines, self.bin_lines):
            hist = FeatureExtractor.get_lbp_histogram(g, b, hist, acc=True)

        mean_val = np.mean(hist)
        if mean_val > 0:
            hist = hist / mean_val

        return hist

    # ------------------------------------------------------------------

    @staticmethod
    def get_lbp_histogram(
        img:  np.ndarray,
        mask: np.ndarray,
        hist: np.ndarray = None,
        acc:  bool = True,
    ) -> np.ndarray:
        """
        Compute (or accumulate) the 256-bin LBP histogram of ``img``
        restricted to the pixels where ``mask`` is non-zero.

        Parameters
        ----------
        img  : H×W uint8 grayscale image
        mask : H×W uint8 binary image (non-zero = include pixel)
        hist : existing histogram array to accumulate into (optional)
        acc  : if True, add to ``hist``; otherwise replace

        Returns
        -------
        hist : ndarray shape (256,)
        """
        height, width = img.shape

        # ── Build LBP image (vectorised) ─────────────────────────────────
        lbp = np.zeros((height, width), dtype=np.uint8)

        v  = 3
        dx = [0,  v,  v,  v,  0, -v, -v, -v]
        dy = [v,  v,  0, -v, -v, -v,  0,  v]

        for i in range(8):
            view_shf = FeatureExtractor.shift(img, ( dy[i],  dx[i]))
            view_img = FeatureExtractor.shift(img, (-dy[i], -dx[i]))
            view_lbp = FeatureExtractor.shift(lbp, (-dy[i], -dx[i]))

            bit_mask  = (view_img >= view_shf).astype(np.uint8)  # 0 or 1
            view_lbp |= (bit_mask << i)

        # ── Mask must be uint8 for cv.calcHist ───────────────────────────
        mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask

        # ── Compute histogram ─────────────────────────────────────────────
        if hist is None or not acc:
            h = cv.calcHist([lbp], [0], mask_u8, [256], [0, 256])
        else:
            init_hist = hist.reshape(256, 1).astype(np.float32)
            h = cv.calcHist(
                [lbp], [0], mask_u8, [256], [0, 256],
                init_hist, True
            )

        return h.ravel().astype(np.float64)

    # ------------------------------------------------------------------

    @staticmethod
    def shift(img: np.ndarray, shift: tuple) -> np.ndarray:
        """
        Return a view of ``img`` shifted by (row_shift, col_shift).

        Positive r shifts down (drops top rows), negative r shifts up.
        Positive c shifts right (drops left cols), negative c shifts left.
        """
        r, c = shift

        if r >= 0:
            img = img[r:,  :]
        else:
            img = img[:r,  :]

        if c >= 0:
            img = img[:,   c:]
        else:
            img = img[:, :c]

        return img
