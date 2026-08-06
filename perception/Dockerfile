ARG PYPI_MIRROR=https://pypi.org/simple/
ARG ROS_BASE_IMAGE=bj-warehouse.tencentcloudcr.com/phanthy-motus/ros-base:latest
FROM ${ROS_BASE_IMAGE}
ARG PYPI_MIRROR

RUN apt-get update && \
    mv /sbin/ldconfig /sbin/ldconfig.real && \
    ln -s /bin/true /sbin/ldconfig && \
    apt-get install -y --no-install-recommends python3-pip && \
    mv /sbin/ldconfig.real /sbin/ldconfig && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url ${PYPI_MIRROR} \
    torch
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    pyyaml requests silero-vad websockets webrtcvad websocket-client dashscope
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    kiss-icp scipy
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    ultralytics
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    sherpa-onnx
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    phonemizer
RUN pip3 install --no-cache-dir -i ${PYPI_MIRROR} \
    onnxruntime opencv-python-headless

# Obstacle models are external artifacts baked into the image at build time
# (downloaded from the juicefs mirror, sha256+size verified) so the runtime
# container never needs network access for them. Destinations live under
# /opt/phanthy-motus/models/obstacle so a runtime /models mount cannot hide them.
COPY models/obstacle-artifacts.json /tmp/obstacle-artifacts.json
COPY utils/obstacle_model_downloader.py /tmp/obstacle_model_downloader.py
RUN python3 /tmp/obstacle_model_downloader.py \
      --manifest /tmp/obstacle-artifacts.json \
      --timeout 120 \
      --retries 3

# Override base image RMW (some ros-base tags ship cyclonedds by default)
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV ROS_DOMAIN_ID=42

WORKDIR /work
COPY main.py    /work/main.py
COPY plugins/   /work/plugins/
COPY utils/     /work/utils/
COPY config.yaml /work/config.yaml
COPY config/fastdds_large_message.xml /opt/phanthy-motus/config/fastdds_large_message.xml
COPY deploy/     /deploy/

EXPOSE 15720
EXPOSE 15721
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && python3 /work/main.py"]
