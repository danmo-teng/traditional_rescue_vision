#!/usr/bin/env python3
from __future__ import annotations

import json
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rescue_vision.config import load_config
from rescue_vision.detector import TraditionalDetector
from rescue_vision.localizer import GroundLocalizer
from rescue_vision.tracker import MultiFrameTracker


def assert_class(detector, frame, class_name):
    detections, debug = detector.detect(frame, [class_name])
    assert detections, f"{class_name} not detected; rejected={[(x.reject_reason, x.score) for x in debug['rejected']]}"
    assert detections[0].class_name == class_name
    return detections


def main() -> None:
    config = load_config(ROOT / "config" / "rescue_vision.json")
    detector = TraditionalDetector(config, GroundLocalizer())

    white = np.full((480, 640, 3), 230, dtype=np.uint8)
    green = white.copy()
    cv2.rectangle(green, (250, 250), (310, 310), (0, 255, 0), -1)
    detections = assert_class(detector, green, "green_supply")

    noisy = green.copy()
    cv2.rectangle(noisy, (30, 30), (34, 34), (0, 255, 0), -1)
    noisy_detections, noisy_debug = detector.detect(noisy, ["green_supply"])
    assert len(noisy_detections) == 1
    valid_mask = noisy_debug["valid_masks"]["green_supply"]
    assert valid_mask[32, 32] == 0, "面积不合格的小色块不应出现在最终白色掩膜"
    assert valid_mask[280, 280] == 255

    # One logical class may contain multiple angle/lighting references. A
    # reference has its own color and geometry rules but keeps the class name.
    angled_reference = copy.deepcopy(config["classes"]["green_supply"])
    angled_reference.pop("references", None)
    angled_reference["reference_name"] = "测试斜视"
    angled_reference["hsv"] = [140, 100, 40, 165, 255, 255]
    angled_reference["lab"] = [0, 0, 0, 255, 255, 255]
    angled_reference["fusion"] = "hsv"
    angled_reference["candidate"]["aspect"] = [2.5, 4.0]
    config["classes"]["green_supply"]["references"] = [angled_reference]
    detector.update_config(config)
    angled = white.copy()
    cv2.rectangle(angled, (220, 250), (400, 310), (255, 0, 255), -1)
    base_only, _ = detector.detect(angled, ["green_supply"], reference_index=0)
    assert not base_only, "调试基础参考时不应混入其他参考的结果"
    reference_only, _ = detector.detect(angled, ["green_supply"], reference_index=1)
    assert reference_only, "应能单独调试指定参考"
    angled_detections = assert_class(detector, angled, "green_supply")
    assert len(angled_detections) == 1, "多个参考命中同一物体后应去重"

    cyan = white.copy()
    cv2.rectangle(cyan, (250, 250), (310, 310), (255, 255, 0), -1)
    assert_class(detector, cyan, "danger_cyan")

    orange = white.copy()
    cv2.rectangle(orange, (230, 250), (350, 310), (0, 128, 255), -1)
    assert_class(detector, orange, "injured_orange")

    black = white.copy()
    cv2.fillConvexPoly(black, np.array([[280, 220], [235, 310], [325, 310]], np.int32), (8, 8, 8))
    assert_class(detector, black, "core_black")

    red = white.copy()
    cv2.rectangle(red, (120, 260), (520, 390), (0, 0, 255), -1)
    assert_class(detector, red, "safe_red")

    blue = white.copy()
    cv2.rectangle(blue, (120, 260), (520, 390), (255, 0, 0), -1)
    assert_class(detector, blue, "safe_blue")

    purple = white.copy()
    cv2.rectangle(purple, (100, 300), (540, 350), (255, 0, 255), -1)
    assert_class(detector, purple, "entrance_purple")

    image_points = [(100, 100), (500, 100), (100, 400), (500, 400)]
    ground_points = [(-300, 400), (300, 400), (-300, 1200), (300, 1200)]
    localizer, rmse = GroundLocalizer.calibrate(image_points, ground_points)
    assert rmse < 0.01
    ground = localizer.image_to_ground((300, 250))
    assert ground is not None and abs(ground[0]) < 0.01 and abs(ground[1] - 800) < 0.01

    tracker = MultiFrameTracker(config)
    tracks = []
    for _ in range(3):
        tracks = tracker.update(detections)
    assert tracker.confirmed(), "three consistent green detections should confirm a track"

    print(json.dumps({"result": "PASS", "confirmed_tracks": len(tracker.confirmed())}))


if __name__ == "__main__":
    main()
