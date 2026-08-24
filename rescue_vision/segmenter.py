from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def odd_kernel(value: int) -> int:
    value = max(0, int(value))
    if value == 0:
        return 0
    return value if value % 2 else value + 1


class Segmenter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def update_config(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def color_spaces(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
            cv2.cvtColor(frame, cv2.COLOR_BGR2LAB),
        )

    @staticmethod
    def _range(image: np.ndarray, bounds: list[int], hue: bool = False) -> np.ndarray:
        low = np.array(bounds[:3], dtype=np.uint8)
        high = np.array(bounds[3:], dtype=np.uint8)
        if hue and bounds[0] > bounds[3]:
            low_a = np.array([bounds[0], bounds[1], bounds[2]], dtype=np.uint8)
            high_a = np.array([179, bounds[4], bounds[5]], dtype=np.uint8)
            low_b = np.array([0, bounds[1], bounds[2]], dtype=np.uint8)
            high_b = np.array([bounds[3], bounds[4], bounds[5]], dtype=np.uint8)
            return cv2.bitwise_or(cv2.inRange(image, low_a, high_a), cv2.inRange(image, low_b, high_b))
        return cv2.inRange(image, low, high)

    def segment(
        self,
        frame: np.ndarray,
        class_name: str,
        hsv: np.ndarray | None = None,
        lab: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        profile = self.config["classes"][class_name]
        return self.segment_profile(frame, profile, hsv, lab)

    def segment_profile(
        self,
        frame: np.ndarray,
        profile: dict[str, Any],
        hsv: np.ndarray | None = None,
        lab: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if hsv is None or lab is None:
            hsv, lab = self.color_spaces(frame)
        hsv_mask = self._range(hsv, profile["hsv"], hue=True)
        lab_mask = self._range(lab, profile["lab"])
        fusion = profile.get("fusion", "and")
        if fusion == "or":
            mask = cv2.bitwise_or(hsv_mask, lab_mask)
        elif fusion == "hsv":
            mask = hsv_mask
        elif fusion == "lab":
            mask = lab_mask
        else:
            mask = cv2.bitwise_and(hsv_mask, lab_mask)

        roi = self.config.get("roi_polygon", [])
        if len(roi) >= 3:
            roi_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [np.asarray(roi, dtype=np.int32)], 255)
            mask = cv2.bitwise_and(mask, roi_mask)

        morph = profile.get("morphology", {})
        open_size = odd_kernel(morph.get("open", 0))
        close_size = odd_kernel(morph.get("close", 0))
        iterations = max(1, int(morph.get("iterations", 1)))
        if open_size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        if close_size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return mask, {"hsv": hsv_mask, "lab": lab_mask, "final": mask}
