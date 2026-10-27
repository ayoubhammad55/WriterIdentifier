"""
Horizontal-histogram-based line segmentor for IAM handwriting forms.

Splits a pre-processed paragraph image into individual text lines.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import DEBUG_LINE_SEGMENTATION

try:
    from utils.utils import display_image
except ImportError:
    display_image = None


class LineSegmentor:

    def __init__(self, gray_img: np.ndarray, bin_img: np.ndarray):
        """
        Parameters
        ----------
        gray_img : ndarray  – grayscale paragraph image (H×W)
        bin_img  : ndarray  – binarized  paragraph image (H×W, 0/255)
        """
        self.gray_img = gray_img
        self.bin_img  = bin_img

        # Guard against empty images
        if gray_img is None or bin_img is None or gray_img.size == 0:
            self._empty = True
            self.peaks  = []
            self.valleys = []
            self.lines_boundaries = []
            return

        self._empty = False

        # Horizontal pixel-density histogram
        self.hor_hist = np.sum(bin_img, axis=1, dtype=int) // 255

        if len(self.hor_hist) == 0:
            self._empty = True
            self.peaks   = []
            self.valleys = []
            self.lines_boundaries = []
            return

        # Thresholds
        self.threshold_high = int(np.max(self.hor_hist) // 3)
        self.threshold_low  = 25

        # Detect structure
        self.peaks           = []
        self.valleys         = []
        self.lines_boundaries = []

        self.detect_peaks()

        if len(self.peaks) == 0:
            # No peaks found -> treat whole image as one line
            self._empty = True
            h, w = bin_img.shape
            self.lines_boundaries = [(0, 0, w - 1, h - 1)]
            return

        self.avg_peaks_dist = max(
            1, int((self.peaks[-1] - self.peaks[0]) // max(len(self.peaks), 1))
        )
        self.detect_valleys()
        self.detect_missing_peaks_valleys()
        self.detect_line_boundaries()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def segment(self):
        """
        Returns
        -------
        (gray_lines, bin_lines) : tuple[list[ndarray], list[ndarray]]
            Lists of cropped line images (gray and binary).
            Returns ([], []) if the image is invalid or has no detectable lines.
        """
        if self._empty and not self.lines_boundaries:
            return [], []

        self.display_segmentation()

        gray_lines, bin_lines = [], []
        for l, u, r, d in self.lines_boundaries:
            g_line = self.gray_img[u:d + 1, l:r + 1]
            b_line = self.bin_img [u:d + 1, l:r + 1]
            if g_line.size > 0:
                gray_lines.append(g_line)
                bin_lines.append(b_line)

        return gray_lines, bin_lines

    # -------------------------------------------------------------------------
    # Peak / valley detection (unchanged logic, hardened against edge cases)
    # -------------------------------------------------------------------------

    def detect_peaks(self):
        self.peaks = []

        i = 0
        while i < len(self.hor_hist):
            if self.hor_hist[i] < self.threshold_high:
                i += 1
                continue

            peak_idx = i
            while i < len(self.hor_hist) and self.is_probable_peak(i):
                if self.hor_hist[i] > self.hor_hist[peak_idx]:
                    peak_idx = i
                i += 1

            self.peaks.append(peak_idx)

    def detect_valleys(self):
        self.valleys = [0]

        i = 1
        while i < len(self.peaks):
            u = self.peaks[i - 1]
            d = self.peaks[i]
            i += 1

            expected_valley = d - self.avg_peaks_dist // 2
            valley_idx = u

            while u < d:
                dist1 = abs(u            - expected_valley)
                dist2 = abs(valley_idx   - expected_valley)

                cond1 = self.hor_hist[u] < self.hor_hist[valley_idx]
                cond2 = (self.hor_hist[u] == self.hor_hist[valley_idx]
                         and dist1 < dist2)

                if cond1 or cond2:
                    valley_idx = u

                u += 1

            self.valleys.append(valley_idx)

        self.valleys.append(len(self.hor_hist) - 1)

    def detect_missing_peaks_valleys(self):
        i     = 1
        found = False

        while i < len(self.valleys):
            up, down = self.valleys[i - 1], self.valleys[i]
            dis = down - up
            i  += 1

            if dis < 1.5 * self.avg_peaks_dist:
                continue

            u = up + self.avg_peaks_dist
            d = min(down, u + self.avg_peaks_dist)

            while (d - u) * 2 > self.avg_peaks_dist:
                if self.is_probable_valley(u) and self.is_probable_valley(d):
                    peak = self.get_peak_in_range(u, d)
                    if self.hor_hist[peak] > self.threshold_low:
                        self.peaks.append(self.get_peak_in_range(u, d))
                        found = True

                u  = u + self.avg_peaks_dist
                d  = min(down, u + self.avg_peaks_dist)

        if found:
            self.peaks.sort()
            self.detect_valleys()

    def detect_line_boundaries(self):
        height, width = self.bin_img.shape
        self.lines_boundaries = []

        i = 1
        while i < len(self.valleys):
            u = self.valleys[i - 1]
            d = self.valleys[i]
            l = 0
            r = width - 1
            i += 1

            while u < d and self.hor_hist[u] == 0:
                u += 1
            while d > u and self.hor_hist[d] == 0:
                d -= 1

            ver_hist = np.sum(self.bin_img[u:d + 1, :], axis=0) // 255

            while l < r and ver_hist[l] == 0:
                l += 1
            while r > l and ver_hist[r] == 0:
                r -= 1

            self.lines_boundaries.append((l, u, r, d))

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def calc_average_line_slope(self) -> int:
        avg_slope = 0
        i = 1
        while i < len(self.valleys):
            u  = self.valleys[i - 1]
            d  = self.valleys[i]
            avg_slope += self.calc_range_slope(u, d)
            i += 1
        return int(avg_slope // max(len(self.valleys) - 1, 1))

    def calc_range_slope(self, up: int, down: int) -> int:
        max_der, min_der = -1e9, 1e9
        while up < down:
            up += 1
            val     = self.hor_hist[up] - self.hor_hist[up - 1]
            max_der = max(max_der, val)
            min_der = min(min_der, val)
        return max_der - min_der

    def get_peak_in_range(self, up: int, down: int) -> int:
        peak_idx = up
        while up < down:
            if self.hor_hist[up] > self.hor_hist[peak_idx]:
                peak_idx = up
            up += 1
        return peak_idx

    def is_probable_peak(self, row: int) -> bool:
        width = 15
        for i in range(-width, width):
            idx = row + i
            if 0 <= idx < len(self.hor_hist):
                if self.hor_hist[idx] >= self.threshold_high:
                    return True
        return False

    def is_probable_valley(self, row: int) -> bool:
        width = 30
        count = 0
        for i in range(-width, width):
            idx = row + i
            if 0 <= idx < len(self.hor_hist):
                if self.hor_hist[idx] <= self.threshold_low:
                    count += 1
            else:
                return True
        return count * 2 >= width

    # -------------------------------------------------------------------------
    # Debugging visualisation
    # -------------------------------------------------------------------------

    def display_segmentation(self):
        if not DEBUG_LINE_SEGMENTATION:
            return

        img = cv.cvtColor(self.gray_img, cv.COLOR_GRAY2BGR)
        for l, u, r, d in self.lines_boundaries:
            cv.rectangle(img, (l, u), (r, d), (0, 0, 255), 2)

        if display_image is not None:
            display_image("Line Segmented Paragraph", img, False)

        plt.figure()
        plt.xlabel("Row index")
        plt.ylabel("Number of black pixels")
        plt.plot(range(len(self.hor_hist)), self.hor_hist)
        plt.plot(
            [0, len(self.hor_hist)],
            [self.threshold_high, self.threshold_high],
            "g--",
        )

        for r in self.peaks:
            plt.plot(r, self.hor_hist[r], "ro")
            plt.plot(
                [r - self.avg_peaks_dist / 2, r + self.avg_peaks_dist / 2],
                [self.hor_hist[r], self.hor_hist[r]],
                "r",
            )
        for r in self.valleys:
            plt.plot(r, self.hor_hist[r], "bs")

        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()
        cv.destroyAllWindows()
