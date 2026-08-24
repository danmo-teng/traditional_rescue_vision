#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  python3-opencv \
  python3-numpy \
  python3-gi \
  gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  v4l-utils

python3 -c 'import cv2, numpy, gi; gi.require_version("Gst", "1.0"); from gi.repository import Gst; print("OpenCV", cv2.__version__, "deployment OK")'

echo "Dependencies installed. Run:"
echo "  python3 web_editor.py --device /dev/video0"
