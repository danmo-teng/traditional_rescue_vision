from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except (ImportError, ValueError) as exc:
    raise RuntimeError(
        "缺少GStreamer Python绑定，请安装python3-gi和gir1.2-gstreamer-1.0"
    ) from exc


@dataclass(frozen=True)
class CameraFrame:
    frame_id: int
    image: np.ndarray
    published_ns: int


class LatestFrameCamera:
    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        Gst.init(None)
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._latest: Optional[CameraFrame] = None
        self._decoded = 0
        self._start_ns = time.monotonic_ns()
        description = (
            f"v4l2src device={device} io-mode=mmap do-timestamp=true ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            "jpegparse ! jpegdec idct-method=ifast ! videoconvert ! video/x-raw,format=BGR ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            "appsink name=vision_sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        self.pipeline = Gst.parse_launch(description)
        self.sink = self.pipeline.get_by_name("vision_sink")
        self.sink.connect("new-sample", self._on_sample)
        self.bus = self.pipeline.get_bus()

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        ok, mapping = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            expected = self.width * self.height * 3
            if mapping.size < expected:
                return Gst.FlowReturn.ERROR
            image = np.frombuffer(mapping.data, np.uint8, expected)
            image = image.reshape(self.height, self.width, 3).copy()
        finally:
            buffer.unmap(mapping)
        with self._lock:
            self._decoded += 1
            self._latest = CameraFrame(self._decoded, image, time.monotonic_ns())
        return Gst.FlowReturn.OK

    def start(self) -> None:
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("摄像头启动失败，设备可能被占用")
        result, _, _ = self.pipeline.get_state(5 * Gst.SECOND)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("摄像头未能在5秒内进入PLAYING状态")

    def latest(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._latest

    def decoded_count(self) -> int:
        with self._lock:
            return self._decoded

    def check_error(self) -> Optional[str]:
        message = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message is None:
            return None
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            return f"GStreamer错误：{error}; {debug}"
        return "摄像头流结束"

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)
