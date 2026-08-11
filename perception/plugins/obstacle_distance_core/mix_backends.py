"""Mix backend: our DA v2 Metric indoor ONNX + zengzhitao's YOLO TRT outdoor.

Indoor keeps the deployed ft head (dav2_indoor_small_ft.onnx, ROI-P1 statistic,
leaderboard indoor F1 = 1.00).  Vehicle frames go through zengzhitao's
TensorRT int8 depth + segmentation path via ``NativeTensorRTSegBackend`` and
the approximate bumper geometry in ``estimator`` (leaderboard outdoor ~0.8).

The depth backend returned here implements both the outdoor metric depth
(``predict_depth``) and the indoor ROI-P1 classifier
(``predict_indoor_distance``) so the estimator routes each domain to the
right head.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import urllib.request
from typing import Mapping

import numpy as np

from .contracts import (
    DepthBackend,
    DepthPrediction,
    ErrorCode,
    InstanceSegmentationBackend,
    ObstacleDistanceError,
    SceneDomain,
)
from .hybrid_tensorrt_backends import _check_deadline, _decode_image, _model_path
from .native_tensorrt_backends import (
    NativeTensorRTSegBackend,
    _NativeTensorRTEngine,
    _YOLO_DEPTH_INPUT_SIZE,
    _prepare_yolo_image,
    _scale_depth_to_original,
)

log = logging.getLogger(__name__)

# Indoor head I/O contract (matches the deployed plugin in 8262c80).
_INDOOR_INPUT_H, _INDOOR_INPUT_W = 308, 308
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_JUICEFS_BASE = os.environ.get(
    "OBSTACLE_MODEL_BASE",
    "http://172.28.4.81:34567/lizhuoju/embodied-ai/obstacle-distance/dav2-metric-small-onnx",
)
_INDOOR_FILES = ("dav2_indoor_small_ft.onnx", "dav2_indoor_small_ft.onnx.data")


def _download(url: str, dst: str) -> None:
    log.info("[obstacle-mix] downloading %s -> %s", url, dst)
    urllib.request.urlretrieve(url, dst)
    log.info(
        "[obstacle-mix] download complete: %s (%.1f MB)",
        dst,
        os.path.getsize(dst) / 1e6,
    )


class MixOnnxTrtDepthBackend:
    """Indoor: our fine-tuned DA v2 ONNX ROI-P1. Vehicle: zengzhitao TRT depth."""

    def __init__(
        self,
        indoor_model_dir: str,
        vehicle_engine: str,
        *,
        ort_threads: int = 4,
    ) -> None:
        self._indoor_dir = indoor_model_dir
        self._vehicle_engine_path = vehicle_engine
        self._ort_threads = ort_threads
        self._session = None
        self._vehicle: _NativeTensorRTEngine | None = None
        self._lock = threading.Lock()

    # ── indoor ────────────────────────────────────────────────────────────

    def _indoor_path(self) -> str:
        onnx = os.path.join(self._indoor_dir, _INDOOR_FILES[0])
        sidecar = onnx + ".data"
        if os.path.isfile(onnx) and os.path.isfile(sidecar):
            return onnx
        os.makedirs(self._indoor_dir, exist_ok=True)
        for fname in _INDOOR_FILES:
            dst = os.path.join(self._indoor_dir, fname)
            if not os.path.isfile(dst):
                _download(f"{_JUICEFS_BASE}/{fname}", dst)
        return onnx

    def _get_session(self):
        if self._session is None:
            with self._lock:
                if self._session is None:
                    import onnxruntime as ort

                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = self._ort_threads
                    opts.inter_op_num_threads = 1
                    self._session = ort.InferenceSession(
                        self._indoor_path(),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    log.info("[obstacle-mix] indoor ONNX session ready")
        return self._session

    def predict_indoor_distance(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> float:
        """ROI-P1 of our ft indoor head; mirrors the deployed statistic."""
        _check_deadline(deadline_monotonic)
        import cv2

        image = _decode_image(image_bytes)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(
            rgb, (_INDOOR_INPUT_W, _INDOOR_INPUT_H), interpolation=cv2.INTER_CUBIC
        )
        x = x.astype(np.float32) / 255.0
        x = (x.transpose(2, 0, 1) - _MEAN) / _STD
        sess = self._get_session()
        depth = sess.run(
            [sess.get_outputs()[0].name],
            {sess.get_inputs()[0].name: x[None].astype(np.float32)},
        )[0]
        depth = np.squeeze(depth).astype(np.float32)
        h, w = image.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # ROI-P1 at the 640x480 reference frame (rows 0-300, cols 213-426).
        r0 = int(round(0 * h / 480))
        r1 = int(round(300 * h / 480))
        c0 = int(round(213 * w / 640))
        c1 = int(round(426 * w / 640))
        roi = depth[r0:r1, c0:c1].astype(np.float64)
        valid = roi[np.isfinite(roi) & (roi > 1e-3)]
        _check_deadline(deadline_monotonic)
        if valid.size == 0:
            return 30.0
        return float(np.percentile(valid, 1.0))

    # ── vehicle ───────────────────────────────────────────────────────────

    def _get_vehicle_engine(self) -> _NativeTensorRTEngine:
        engine = self._vehicle
        if engine is None:
            with self._lock:
                if self._vehicle is None:
                    started = time.monotonic()
                    vehicle = _NativeTensorRTEngine(
                        self._vehicle_engine_path,
                        expected_task="depth",
                    )
                    if vehicle.input_shape != (
                        1,
                        3,
                        _YOLO_DEPTH_INPUT_SIZE,
                        _YOLO_DEPTH_INPUT_SIZE,
                    ):
                        raise ObstacleDistanceError(
                            ErrorCode.MODEL_ERROR,
                            "YOLO depth TensorRT engine input shape is incompatible",
                        )
                    log.info(
                        "[obstacle-mix] YOLO depth engine loaded in %.1fms",
                        1000.0 * (time.monotonic() - started),
                    )
                    self._vehicle = vehicle
                engine = self._vehicle
        return engine

    def predict_depth(
        self,
        image_bytes: bytes,
        domain: SceneDomain,
        deadline_monotonic: float,
    ) -> DepthPrediction:
        if domain is not SceneDomain.VEHICLE:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "mix backend indoor inference requires predict_indoor_distance",
            )
        _check_deadline(deadline_monotonic)
        image = _decode_image(image_bytes)
        height, width = image.shape[:2]
        tensor, _, _, _ = _prepare_yolo_image(image, _YOLO_DEPTH_INPUT_SIZE)
        outputs = self._get_vehicle_engine().infer(tensor)
        if len(outputs) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "YOLO depth TensorRT engine must have one output",
            )
        depth = _scale_depth_to_original(outputs[0].squeeze(), height, width)
        _check_deadline(deadline_monotonic)
        return DepthPrediction(
            depth_m=np.ascontiguousarray(depth, dtype=np.float32),
            source_height=height,
            source_width=width,
        )


def create_backends(
    config: Mapping,
) -> tuple[DepthBackend, InstanceSegmentationBackend]:
    indoor_model_dir = config.get(
        "indoor_model_dir", "/opt/phanthy-motus/models/obstacle"
    )
    vehicle_engine = _model_path(config, "vehicle_depth_engine")
    segmentation_engine = _model_path(config, "segmentation_engine")
    vehicle_config = config.get("vehicle", {})
    if not isinstance(vehicle_config, Mapping):
        vehicle_config = {}
    return (
        MixOnnxTrtDepthBackend(
            indoor_model_dir,
            vehicle_engine,
            ort_threads=int(os.environ.get("OBSTACLE_ORT_THREADS", "4")),
        ),
        NativeTensorRTSegBackend(
            segmentation_engine,
            allowed_classes=vehicle_config.get("allowed_classes"),
            min_confidence=vehicle_config.get("min_confidence"),
        ),
    )
