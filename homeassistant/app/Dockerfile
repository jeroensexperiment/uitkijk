# Uitkijk as a Home Assistant app: the same service as the main Dockerfile, plus go2rtc and the
# Google Coral (driver, LiteRT and a YOLO11n model compiled for the Edge TPU).
# Home Assistant builds this folder on its own, so the source comes from GitHub (build.yaml).

# --- source ------------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS src
ARG UITKIJK_REPO=https://github.com/jeroensexperiment/uitkijk.git
ARG UITKIJK_REF=main
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates \
 && git clone --depth 1 --branch "$UITKIJK_REF" "$UITKIJK_REPO" /src

# --- object models ---------------------------------------------------------------------------
# CPU: YOLO11n, converted once by .github/workflows/models.yml (release "models"; converting here
# took half an hour). Coral: Google's ready-made SSDLite MobileDet for the Edge TPU (also what
# Frigate uses); a YOLO model for the Edge TPU needs a compiler that is no longer published.
FROM python:3.11-slim-bookworm AS model
ARG MODELS_URL=https://github.com/jeroensexperiment/uitkijk/releases/download/models
ARG CORAL_MODEL_URL=https://github.com/google-coral/test_data/raw/master/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates \
 && mkdir -p /out \
 && curl -fsSL -o /out/yolo11n.onnx "$MODELS_URL/yolo11n.onnx" \
 && curl -fsSL -o /out/ssdlite_mobiledet_edgetpu.tflite "$CORAL_MODEL_URL"

# --- the app -------------------------------------------------------------------------------------
# fixed base: the Supervisor would otherwise use its own (Alpine) base image
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CAMSVC_CONFIG=/config/config.yaml CAMSVC_STATE=/data/state CAMSVC_CLIPS=/media/uitkijk \
    CAMSVC_MODEL=/data/models/yolo11n.onnx CAMSVC_BUNDLED_MODELS=/opt/camsvc-models \
    CAMSVC_CORAL_MODEL=/opt/camsvc-models/ssdlite_mobiledet_edgetpu.tflite HOME=/data
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
