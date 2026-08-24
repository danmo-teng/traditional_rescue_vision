#!/usr/bin/env python3
"""Headless runtime with tiered detection rates and latest-frame processing."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from rescue_vision.camera import LatestFrameCamera
from rescue_vision.config import load_config
from rescue_vision.detector import TraditionalDetector
from rescue_vision.localizer import GroundLocalizer
from rescue_vision.tracker import MultiFrameTracker


def track_dict(track, namespace: str) -> dict:
    detection = track.last_detection
    return {
        "id": f"{namespace}{track.track_id}",
        "class": track.class_name,
        "confidence": round(track.confidence, 4),
        "state": track.state.value,
        "position": [round(value, 2) for value in track.position],
        "coordinate_system": "ground_mm" if detection.ground_xy_mm is not None else "image_px",
        "size_mm": None if detection.size_mm is None else [round(value, 2) for value in detection.size_mm],
        "hits": track.hits,
        "misses": track.misses,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--camera-fps", type=int, default=350)
    parser.add_argument("--config", default=str(root / "config" / "rescue_vision.json"))
    parser.add_argument("--homography", default=str(root / "config" / "homography.txt"))
    parser.add_argument("--output", default=str(root / "runtime_result.json"))
    parser.add_argument("--print-fps", type=float, default=2.0)
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = config["runtime"]
    danger_classes = runtime.get("danger_classes", ["danger_cyan"])
    materials = [name for name in runtime["material_classes"] if name not in danger_classes]
    zones = runtime["zone_classes"]
    danger_period = 1.0 / float(runtime.get("danger_fps", 120))
    material_period = 1.0 / float(runtime.get("material_fps", 60))
    zone_period = 1.0 / float(runtime.get("zone_fps", 20))
    localizer = GroundLocalizer.load(args.homography)
    detector = TraditionalDetector(config, localizer)
    danger_tracker = MultiFrameTracker(config)
    material_tracker = MultiFrameTracker(config)
    zone_tracker = MultiFrameTracker(config)
    camera = LatestFrameCamera(args.device, 640, 480, args.camera_fps)
    output = Path(args.output)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    camera.start()
    next_danger = next_material = next_zone = next_output = next_print = time.perf_counter()
    last_danger_id = last_material_id = last_zone_id = 0
    danger_cost_ms = material_cost_ms = zone_cost_ms = 0.0

    try:
        while running:
            packet = camera.latest()
            if packet is None:
                time.sleep(0.001)
                continue
            now = time.perf_counter()
            material_due = now >= next_material and packet.frame_id != last_material_id
            danger_due = now >= next_danger and packet.frame_id != last_danger_id
            if material_due:
                # At each 60 Hz full-material tick, reuse the same HSV/Lab
                # conversion for danger and ordinary materials. The alternating
                # ticks only run the latency-critical danger class.
                next_material = now + material_period
                next_danger = now + danger_period
                started = time.perf_counter()
                detections, _ = detector.detect(packet.image, danger_classes + materials)
                danger_set = set(danger_classes)
                danger_tracker.update([item for item in detections if item.class_name in danger_set])
                material_tracker.update([item for item in detections if item.class_name not in danger_set])
                material_cost_ms = (time.perf_counter() - started) * 1000.0
                danger_cost_ms = material_cost_ms
                last_danger_id = last_material_id = packet.frame_id
            elif danger_due:
                next_danger = now + danger_period
                started = time.perf_counter()
                detections, _ = detector.detect(packet.image, danger_classes)
                danger_tracker.update(detections)
                danger_cost_ms = (time.perf_counter() - started) * 1000.0
                last_danger_id = packet.frame_id
            if now >= next_zone and packet.frame_id != last_zone_id:
                next_zone = now + zone_period
                started = time.perf_counter()
                detections, _ = detector.detect(packet.image, zones)
                zone_tracker.update(detections)
                zone_cost_ms = (time.perf_counter() - started) * 1000.0
                last_zone_id = packet.frame_id
            if now >= next_output:
                next_output = now + 0.02
                danger_tracks = danger_tracker.confirmed()
                material_tracks = material_tracker.confirmed()
                zone_tracks = zone_tracker.confirmed()
                result = {
                    "timestamp_monotonic_ns": time.monotonic_ns(),
                    "frame_id": packet.frame_id,
                    "frame_age_ms": round((time.monotonic_ns() - packet.published_ns) / 1_000_000.0, 3),
                    "calibrated": localizer.calibrated,
                    "tracks": [track_dict(track, "D") for track in danger_tracks]
                    + [track_dict(track, "M") for track in material_tracks]
                    + [track_dict(track, "Z") for track in zone_tracks],
                }
                temporary = output.with_suffix(output.suffix + ".tmp")
                temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                temporary.replace(output)
            if now >= next_print:
                next_print = now + 1.0 / args.print_fps
                confirmed = len(danger_tracker.confirmed()) + len(material_tracker.confirmed()) + len(zone_tracker.confirmed())
                print(f"frame={packet.frame_id} danger={danger_cost_ms:.2f}ms material={material_cost_ms:.2f}ms zone={zone_cost_ms:.2f}ms confirmed={confirmed}")
            time.sleep(0.0005)
    finally:
        camera.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
