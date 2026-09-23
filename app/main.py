"""FastAPI entrypoint for the POPF local AI recognition MVP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.feedback import append_feedback
from app.inference import (
    InferenceInputError,
    InvalidImageError,
    PredictionService,
    UnsupportedImageFormatError,
)
from app.knowledge import KnowledgeIndex


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOG_PATH = PROJECT_ROOT / "artifacts/logs/popf_web.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("popf.web")

app = FastAPI(
    title="POPF 京剧脸谱 AI 识别",
    version="2026 Reconstruction MVP",
    docs_url="/api/docs",
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_service: PredictionService | None = None
_service_error: Exception | None = None
_knowledge = KnowledgeIndex()


class FeedbackRequest(BaseModel):
    upload_id: str = Field(min_length=1, max_length=100)
    top1: dict[str, Any]
    top3: list[dict[str, Any]]
    top1_correct: bool
    corrected_label: str | None = None


def _load_service() -> PredictionService:
    global _service, _service_error
    if _service is not None:
        return _service
    try:
        _service = PredictionService()
        _service_error = None
        logger.info("Loaded model checkpoint: %s", _service.checkpoint_path)
        return _service
    except Exception as exc:
        _service_error = exc
        logger.exception("Model loading failed")
        raise


@app.on_event("startup")
def load_model_at_startup() -> None:
    try:
        _load_service()
    except Exception:
        # Keep the HTTP process alive so /health can explain the failure
        # without exposing a traceback to the browser.
        pass


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    if _service is None:
        try:
            service = _load_service()
        except Exception:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "model": "resnet18_conservative_finetune",
                    "message": "模型暂时无法加载，请检查本地 checkpoint 和运行环境。",
                },
            )
    else:
        service = _service
    return JSONResponse(content=service.health())


@app.post("/api/predict")
async def predict(file: UploadFile | None = File(default=None)) -> dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="请先上传一张图片。")
    try:
        image_bytes = await file.read()
        service = _load_service()
        result = service.predict(image_bytes, file.filename or "", top_k=3)
        top1_label = result["top1"]["label"]
        result["knowledge"] = _knowledge.knowledge_for_label(top1_label)
        result["related_images"] = _knowledge.related_images(top1_label)
        logger.info(
            "Prediction completed upload_id=%s top1=%s",
            result["upload_id"],
            top1_label,
        )
        return result
    except UnsupportedImageFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail="当前图片暂时无法完成识别，请更换图片后重试。",
        ) from exc


@app.get("/api/reference-image/{image_id}")
def reference_image(image_id: str) -> Response:
    try:
        data, media_type = _knowledge.image_bytes(image_id)
    except (KeyError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="相关图片暂时不可用。") from exc
    return Response(content=data, media_type=media_type)


@app.post("/api/feedback")
def feedback(request: FeedbackRequest) -> dict[str, Any]:
    try:
        service = _load_service()
        return append_feedback(
            upload_id=request.upload_id,
            top1=request.top1,
            top3=request.top3,
            top1_correct=request.top1_correct,
            corrected_label=request.corrected_label,
            allowed_labels=set(service.class_names),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feedback persistence failed")
        raise HTTPException(
            status_code=500,
            detail="反馈暂时无法保存，请稍后重试。",
        ) from exc

