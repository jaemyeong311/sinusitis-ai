"""Sinusitis CorrectFlip inference API. Run: uvicorn main:app"""
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import os
import secrets
import threading
import warnings
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from model_definition_lightcnn_v2 import SinusitisLightCNN, CLASS_ORDER

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
LABELS_KO = ['정상', '왼쪽', '오른쪽', '양쪽']
MODEL_VERSION = 'CorrectFlip-LightCNN-v2-best-val-loss'
lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS', '1')))
    checkpoint = torch.load(ROOT / 'best_val_loss.pt', map_location='cpu', weights_only=True)
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model = SinusitisLightCNN()
    model.load_state_dict(state, strict=True)
    model.eval()
    app.state.model = model
    yield
    del app.state.model

app = FastAPI(title='Sinusitis AI API', version='1.0.0', lifespan=lifespan)

@app.get('/health')
def health():
    return {'status': 'ok', 'modelVersion': MODEL_VERSION, 'classes': CLASS_ORDER}

@app.post('/predict')
def predict(image: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)):
    try:
        expected = os.getenv('AI_API_KEY', '')
        if expected and not secrets.compare_digest((x_api_key or '').encode(), expected.encode()):
            raise HTTPException(401, 'Invalid API key')
        if image.content_type not in ('image/jpeg', 'image/png'):
            raise HTTPException(415, 'Only JPEG and PNG images are supported')
        data = image.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, 'Image must be 5 MiB or smaller')
        if not data:
            raise HTTPException(400, 'Empty image')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as source:
                    if source.format not in ('JPEG', 'PNG'):
                        raise HTTPException(415, 'Only JPEG and PNG images are supported')
                    if source.width * source.height > MAX_PIXELS:
                        raise HTTPException(413, 'Image exceeds 16 million pixels')
                    # Match supplied example exactly. No mirror, EXIF rotation or normalization.
                    rgb = source.convert('RGB').resize((96, 96), Image.Resampling.BICUBIC)
                    array = np.asarray(rgb, dtype=np.float32) / 255.0
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(413, 'Image dimensions are too large')
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(400, 'Invalid or damaged image')
        x = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
        with lock, torch.inference_mode():
            probabilities = torch.softmax(app.state.model(x), dim=1)[0].tolist()
        index = int(np.argmax(probabilities))
        return {
            'classIndex': index,
            'predictedLabel': CLASS_ORDER[index],
            'predictedLabelKo': LABELS_KO[index],
            'confidence': probabilities[index],
            'probabilities': dict(zip(CLASS_ORDER, probabilities)),
            'modelVersion': MODEL_VERSION,
        }
    finally:
        image.file.close()
