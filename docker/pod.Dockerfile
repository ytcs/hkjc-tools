# GPU pod image: the RunPod base the pods already use + this repo's locked environment (uv.lock, vision group)
# and the COCO detector weights, so a pod starts without downloading ~5 GB from PyPI. The code itself is cloned
# at a tag on the pod; `uv sync --frozen` there only installs the project into /opt/hkjc-venv.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
ENV UV_PROJECT_ENVIRONMENT=/opt/hkjc-venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=automatic
RUN pip install --no-cache-dir uv==0.13.7
COPY pyproject.toml uv.lock .python-version /tmp/build/
RUN cd /tmp/build && uv sync --group vision --frozen --no-install-project && rm -rf /root/.cache/uv /tmp/build
RUN mkdir -p /opt/hkjc-models && python3 -c "import urllib.request as u; u.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt', '/opt/hkjc-models/coco_yolov8x.pt')"
LABEL org.opencontainers.image.source=https://github.com/ytcs/hkjc-tools
