# GPU pod image: the RunPod base the pods already use + this repo's locked environment (uv.lock, vision group)
# and the COCO detector weights, so a pod starts without downloading ~5 GB from PyPI. The code itself is cloned
# at a tag on the pod; `uv sync --frozen` there only installs the project into /opt/hkjc-venv.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
ENV UV_PROJECT_ENVIRONMENT=/opt/hkjc-venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=automatic
RUN pip install --no-cache-dir uv==0.13.7
COPY pyproject.toml uv.lock .python-version /tmp/build/
RUN cd /tmp/build && uv sync --group vision --frozen --no-install-project && rm -rf /root/.cache/uv /tmp/build
# CUDA OCR (journal 0216, arm G): the onnxruntime-gpu wheel matching the locked onnxruntime, installed in place of it
# at pod setup (offline); the lock keeps onnxruntime so local CPU runs are unchanged
RUN V=$(/opt/hkjc-venv/bin/python -c "import onnxruntime as o; print(o.__version__)") && \
    pip download --no-cache-dir --no-deps --only-binary=:all: --python-version 3.11 --implementation cp \
      --abi cp311 --platform manylinux_2_28_x86_64 "onnxruntime-gpu==$V" -d /opt/hkjc-wheels
RUN mkdir -p /opt/hkjc-models && python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt', '/opt/hkjc-models/coco_yolov8x.pt')"
LABEL org.opencontainers.image.source=https://github.com/ytcs/hkjc-tools
