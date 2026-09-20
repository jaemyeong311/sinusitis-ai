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
from PIL import Image, UnidentifiedImageError

from gradcam_lightcnn_v1 import (
    GradCAM,
    load_checkpoint_model,
    preprocess_image,
)
from model_definition_lightcnn_v2 import CLASS_ORDER


# ========================================
# 1. 기본 설정
# ========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_val_loss.pt"

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000

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

        # 업로드 파일 형식 검사
        if image.content_type not in (
            "image/jpeg",
            "image/png",
        ):
            raise HTTPException(
                status_code=415,
                detail="JPG 또는 PNG 파일만 업로드해주세요.",
            )

        # 파일 크기 검사
        image_bytes = image.file.read(
            MAX_FILE_SIZE + 1
        )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="빈 이미지 파일입니다.",
            )

        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="이미지는 5MB 이하로 업로드해주세요.",
            )

        # 실제 이미지 검증 및 전처리
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    Image.DecompressionBombWarning,
                )

                with Image.open(
                    BytesIO(image_bytes)
                ) as source:
                    if source.format not in ("JPEG", "PNG"):
                        raise HTTPException(
                            status_code=415,
                            detail="실제 파일 형식이 JPG 또는 PNG가 아닙니다.",
                        )

                    if (
                        source.width * source.height
                        > MAX_IMAGE_PIXELS
                    ):
                        raise HTTPException(
                            status_code=413,
                            detail="이미지는 1,600만 픽셀 이하여야 합니다.",
                        )

                    source.verify()

                # 첨부된 Grad-CAM 패키지의 전처리를 그대로 사용
                # RGB → 96x96 BILINEAR → float32 / 255
                # 좌우 반전은 적용하지 않음
                rgb, tensor = preprocess_image(
                    BytesIO(image_bytes)
                )

        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            raise HTTPException(
                status_code=413,
                detail="이미지 해상도가 너무 큽니다.",
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
        ):
            raise HTTPException(
                status_code=400,
                detail="이미지를 읽을 수 없거나 파일이 손상됐습니다.",
            )

        # Grad-CAM 계산
        # Grad-CAM에는 gradient가 필요하므로
        # torch.no_grad() 또는 inference_mode()를 사용하지 않음
        with MODEL_LOCK, torch.enable_grad():
            gradcam = GradCAM(app.state.model)

            try:
                (
                    heatmap,
                    probabilities,
                    predicted,
                    target,
                ) = gradcam(tensor)

            finally:
                # 요청 종료 시 hook과 gradient 정리
                gradcam.close()
                app.state.model.zero_grad(
                    set_to_none=True
                )

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