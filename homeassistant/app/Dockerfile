# Uitkijk as a Home Assistant app: the same service as the main Dockerfile, plus go2rtc and the
# Google Coral (driver, LiteRT and a YOLO11n model compiled for the Edge TPU).
# Home Assistant builds this folder on its own, so the source comes from GitHub (build.yaml).

# --- source ------------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS src
ARG UITKIJK_REPO=https://github.com/lexi1986/uitkijk.git
ARG UITKIJK_REF=main
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates \
 && git clone --depth 1 --branch "$UITKIJK_REF" "$UITKIJK_REPO" /src

# --- YOLO11n to ONNX (CPU) and to the Edge TPU (Coral) --------------------------------------
FROM python:3.11-slim-bookworm AS model
WORKDIR /m
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 libxcb1 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir ultralytics onnx onnxslim \
 && yolo export model=yolo11n.pt format=onnx imgsz=320 opset=17 simplify=True \
 && mkdir -p /out && cp yolo11n.onnx /out/
# Google's package source for the Edge TPU compiler is gone; the compiler with its own libraries
# is still in Google's GitHub repository. A failed Coral build is not fatal: then the CPU is used.
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/google-coral/edgetpu /edgetpu \
 && git -C /edgetpu sparse-checkout set compiler/x86_64
ENV PATH=/edgetpu/compiler/x86_64:$PATH
RUN (yolo export model=yolo11n.pt format=edgetpu imgsz=320 \
     && cp "$(find /m -name '*_edgetpu.tflite' | head -1)" /out/yolo11n_edgetpu.tflite) \
    || echo "Note: building the Coral model failed; the app will use the CPU"

# --- the app -------------------------------------------------------------------------------------
# fixed base: the Supervisor would otherwise use its own (Alpine) base image
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CAMSVC_CONFIG=/data/config.yaml CAMSVC_STATE=/data/state CAMSVC_CLIPS=/media/uitkijk \
    CAMSVC_MODEL=/data/models/yolo11n.onnx CAMSVC_BUNDLED_MODELS=/opt/camsvc-models \
    CAMSVC_CORAL_MODEL=/opt/camsvc-models/yolo11n_edgetpu.tflite HOME=/data
ARG LIBEDGETPU=https://github.com/feranick/libedgetpu/releases/download/16.0TF2.19.1-1/libedgetpu1-std_16.0tf2.19.1-1.bookworm_amd64.deb
# libglib for OpenCV; VA-API for hardware decoding on Intel; libedgetpu (community build) for the Coral
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends libglib2.0-0 ca-certificates curl libusb-1.0-0 \
    intel-media-va-driver vainfo \
 && curl -fsSL -o /tmp/libedgetpu.deb "$LIBEDGETPU" && apt-get install -y -qq /tmp/libedgetpu.deb \
 && rm -rf /tmp/libedgetpu.deb /var/lib/apt/lists/*
COPY --from=alexxit/go2rtc:1.9.9 /usr/local/bin/go2rtc /usr/local/bin/go2rtc
WORKDIR /opt/camera-service
COPY --from=src /src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ai-edge-litert==2.3.0
COPY --from=model /out/ /opt/camsvc-models/
COPY --from=src /src/camsvc ./camsvc
COPY --from=src /src/config.example.yaml ./
COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
