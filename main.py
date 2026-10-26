import base64
import os
import secrets
import threading
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from media_frames import MediaError, extract_frames

from gradcam_lightcnn_v1 import (
    GradCAM,
    load_checkpoint_model,
)
from model_definition_lightcnn_v2 import CLASS_ORDER


# ========================================
# 1. 기본 설정
# ========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_val_loss.pt"

MAX_FILE_SIZE = 30 * 1024 * 1024
MAX_STILL_IMAGE_SIZE = 5 * 1024 * 1024

MODEL_VERSION = "CorrectFlip-LightCNN-v2-GradCAM-v1-bilinear"

LABELS_KO = ["정상", "왼쪽", "오른쪽", "양쪽"]

# 동시에 들어온 요청이 Grad-CAM 계산을 방해하지 않도록 보호
MODEL_LOCK = threading.Lock()


# ========================================
# 2. 서버 시작 시 모델 로딩
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(
        int(os.getenv("TORCH_NUM_THREADS", "1"))
    )

    model, _ = load_checkpoint_model(
        MODEL_PATH,
        device="cpu",
    )

    app.state.model = model

    print("모델 로딩 완료: best_val_loss.pt")
    print("Grad-CAM API 준비 완료")

    yield

    del app.state.model


app = FastAPI(
    title="Sinusitis Grad-CAM API",
    description="부비동염 분류 및 Grad-CAM 이미지 생성 API",
    version="2.0.0",
    lifespan=lifespan,
)


# ========================================
# 3. 이미지 배열을 PNG Base64로 변환
# ========================================

def to_png_data_url(rgb):
    pixels = (
        np.clip(rgb, 0, 1) * 255
    ).round().astype(np.uint8)

    buffer = BytesIO()

    Image.fromarray(pixels).save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return "data:image/png;base64," + encoded


# ========================================
# 4. 서버 상태 확인
# ========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelVersion": MODEL_VERSION,
        "classes": CLASS_ORDER,
        "gradcamEnabled": True,
    }


# ========================================
# 5. 이미지 예측 및 Grad-CAM 생성
# ========================================

@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    try:
        # 선택적 API 키 인증
        expected_key = os.getenv("AI_API_KEY", "")

        if expected_key:
            supplied_key = x_api_key or ""

            if not secrets.compare_digest(
                supplied_key.encode(),
                expected_key.encode(),
            ):
                raise HTTPException(
                    status_code=401,
                    detail="API 키가 올바르지 않습니다.",
                )

        # 업로드 형식 및 크기 확인
        if image.content_type not in (
            "image/jpeg", "image/png", "image/gif", "video/mp4",
        ):
            raise HTTPException(
                status_code=415,
                detail="JPG, PNG, GIF, MP4 파일만 업로드해주세요.",
            )

        image_bytes = image.file.read(MAX_FILE_SIZE + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="GIF/MP4 파일은 30MB 이하여야 합니다.",
            )
        if (
            image.content_type in ("image/jpeg", "image/png")
            and len(image_bytes) > MAX_STILL_IMAGE_SIZE
        ):
            raise HTTPException(
                status_code=413,
                detail="JPG/PNG 이미지는 5MB 이하여야 합니다.",
            )

        # 영상은 최대 다섯 프레임을 샘플링한다. 가운데 샘플을
        # 기존 단일 이미지 API 결과/Grad-CAM의 기준으로 사용한다.
        try:
            frames, source_frame_count = extract_frames(
                image_bytes, image.content_type
            )
        except MediaError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=str(exc)
            ) from exc

        selected_offset = len(frames) // 2
        selected_frame = frames[selected_offset]
        frame_results = []

        # 하나의 모델을 여러 요청이 동시에 사용하지 않도록 보호
        with MODEL_LOCK:
            for offset, frame in enumerate(frames):
                rgb_frame = (
                    np.asarray(frame.image, dtype=np.float32) / 255.0
                )
                tensor = torch.from_numpy(
                    rgb_frame.transpose(2, 0, 1)
                ).unsqueeze(0)

                if offset == selected_offset:
                    # 선택한 프레임에서만 Grad-CAM 생성
                    with torch.enable_grad():
                        gradcam = GradCAM(app.state.model)
                        try:
                            (
                                heatmap,
                                frame_probs,
                                frame_predicted,
                                target,
                            ) = gradcam(tensor)
                        finally:
                            gradcam.close()
                            app.state.model.zero_grad(
                                set_to_none=True
                            )
                    rgb = rgb_frame
                    probabilities = frame_probs
                    predicted = frame_predicted
                else:
                    with torch.no_grad():
                        frame_probs = torch.softmax(
                            app.state.model(tensor), dim=1
                        )[0].cpu().numpy()
                    frame_predicted = int(np.argmax(frame_probs))

                if not np.isfinite(frame_probs).all():
                    raise HTTPException(
                        status_code=500,
                        detail="모델 계산 결과가 올바르지 않습니다.",
                    )

                frame_results.append({
                    "frameIndex": frame.index,
                    "timeSeconds": frame.time_seconds,
                    "classIndex": int(frame_predicted),
                    "predictedLabel": CLASS_ORDER[frame_predicted],
                    "confidence": float(frame_probs[frame_predicted]),
                    "probabilities": {
                        label: float(frame_probs[index])
                        for index, label in enumerate(CLASS_ORDER)
                    },
                })

        # 계산 결과 확인
        if (
            not np.isfinite(heatmap).all()
            or not np.isfinite(probabilities).all()
        ):
            raise HTTPException(
                status_code=500,
                detail="모델 계산 결과가 올바르지 않습니다.",
            )

        # 히트맵에 색상 적용
        colored_heatmap = matplotlib.colormaps["jet"](
            heatmap
        )[..., :3]

        informative = bool(
            float(heatmap.max() - heatmap.min()) > 1e-12
        )

        # 모델 입력 이미지와 히트맵을 합성
        if informative:
            overlay = np.clip(
                0.6 * rgb + 0.4 * colored_heatmap,
                0,
                1,
            )
        else:
            # 구분되는 강조 영역이 없으면 입력 이미지만 반환
            overlay = rgb

        # 예측 결과와 Grad-CAM 이미지 반환
        return {
            "classIndex": int(predicted),
            "predictedLabel": CLASS_ORDER[predicted],
            "predictedLabelKo": LABELS_KO[predicted],
            "confidence": float(
                probabilities[predicted]
            ),
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(CLASS_ORDER)
            },
            "modelVersion": MODEL_VERSION,
            "analysisFrame": {
                "frameIndex": selected_frame.index,
                "timeSeconds": selected_frame.time_seconds,
                "sampledFrames": len(frames),
                "sourceFrameCount": source_frame_count,
            },
            "frameResults": frame_results,

            "gradcam": {
                "targetClassIndex": int(target),
                "targetLabel": CLASS_ORDER[target],
                "targetLayer": "f.8 (last Conv2d)",
                "width": 96,
                "height": 96,
                "informative": informative,

                "inputImage": to_png_data_url(rgb),
                "heatmapImage": to_png_data_url(
                    colored_heatmap
                ),
                "overlayImage": to_png_data_url(overlay),
            },
        }

    finally:
        image.file.close()