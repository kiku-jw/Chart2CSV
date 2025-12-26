"""
Chart2CSV API - Extract data from chart images.

Production API service for chart data extraction.
"""

import os
import io
import time
import base64
import hashlib
import logging
from typing import Optional
from contextlib import asynccontextmanager

import uuid

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Rate limiting
from collections import defaultdict
from datetime import datetime, timedelta

# Import chart2csv core
from chart2csv.core.pipeline import extract_chart
from chart2csv.core.types import ChartType, Scale
from chart2csv.core.llm_extraction import extract_chart_llm, llm_result_to_csv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

REQUESTS_TOTAL = Counter('chart2csv_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('chart2csv_request_latency_seconds', 'Request latency', ['endpoint'])
EXTRACTIONS_TOTAL = Counter('chart2csv_extractions_total', 'Total extractions', ['mode', 'chart_type'])
ACTIVE_REQUESTS = Gauge('chart2csv_active_requests', 'Active requests')


# --- Models ---

class ExtractionResult(BaseModel):
    """Response model for chart extraction."""
    success: bool
    chart_type: str
    confidence: float
    data: list[dict]
    csv: str
    warnings: list[str] = []
    processing_time_ms: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    code: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float


class APIResponse(BaseModel):
    """Unified API response wrapper."""
    success: bool
    data: dict | None = None
    error: dict | None = None
    meta: dict | None = None


# --- Rate Limiting ---

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[key] = [t for t in self.requests[key] if t > minute_ago]
        
        if len(self.requests[key]) >= self.requests_per_minute:
            return False
        
        self.requests[key].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=20)
start_time = time.time()


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Chart2CSV API starting", extra={
        "version": "1.0.0",
        "environment": os.environ.get("ENV", "production")
    })
    yield
    logger.info("Chart2CSV API shutting down")


app = FastAPI(
    title="Chart2CSV API",
    description="Extract data from chart images using AI. Supports line charts, bar charts, and scatter plots.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# API v1 router
v1_router = APIRouter(prefix="/v1", tags=["v1"])


# Request ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracking."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)

# CORS - Configure allowed origins from environment
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    # Default allowed origins for production
    ALLOWED_ORIGINS = [
        "https://kiku-jw.github.io",
        "https://kikuai-lab.github.io",
        "https://chart2csv.kikuai.dev",
        "https://kikuai.dev",
        "https://www.kikuai.dev",
        "http://localhost:3000",  # Development frontend
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Forwarded-For"],
)


# --- Helpers ---

def get_client_ip(x_forwarded_for: Optional[str] = Header(None)) -> str:
    """Extract client IP for rate limiting."""
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return "unknown"


def image_to_temp_path(image_bytes: bytes) -> str:
    """
    Save image bytes to temp file with security validation.

    Raises:
        ValueError: If image format is invalid or dimensions too large
    """
    import tempfile

    # Security: Set maximum image size to prevent decompression bombs
    MAX_IMAGE_PIXELS = 89478485  # PIL default (about 8192x10922)
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    # Detect format and validate
    img = Image.open(io.BytesIO(image_bytes))

    # Security: Validate image format
    if img.format not in ['PNG', 'JPEG', 'WEBP']:
        raise ValueError(f"Unsupported image format: {img.format}. Only PNG, JPEG, and WEBP are allowed.")

    # Security: Check image dimensions to prevent decompression bombs
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image too large: {img.width}x{img.height} pixels. Maximum is {MAX_IMAGE_PIXELS} pixels.")

    # Additional size check for reasonable chart dimensions
    if img.width > 10000 or img.height > 10000:
        raise ValueError(f"Image dimensions too large: {img.width}x{img.height}. Maximum is 10000x10000.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        return f.name


def parse_csv_to_data(csv_content: str) -> list[dict]:
    """Parse CSV string to list of dicts."""
    lines = csv_content.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = [h.strip() for h in lines[0].split(",")]
    data = []

    for line in lines[1:]:
        values = [v.strip() for v in line.split(",")]
        if len(values) == len(headers):
            row = {}
            for i, h in enumerate(headers):
                try:
                    row[h] = float(values[i])
                except ValueError:
                    row[h] = values[i]
            data.append(row)

    return data


async def _process_chart_extraction(
    image_bytes: bytes,
    mode: str = "llm",
    chart_type: Optional[str] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    calibration_points: Optional[dict] = None,
    use_mistral: bool = True
) -> ExtractionResult:
    """
    Core extraction logic shared across all endpoints.

    Args:
        image_bytes: Raw image bytes
        mode: Extraction mode (llm, cv, auto)
        chart_type: Optional chart type override
        x_scale: X-axis scale (linear, log)
        y_scale: Y-axis scale (linear, log)
        calibration_points: Optional manual calibration data
        use_mistral: Whether to use Mistral OCR

    Returns:
        ExtractionResult with extracted data
    """
    import asyncio

    start = time.time()
    temp_path = None

    try:
        # Save to temp file with validation
        temp_path = image_to_temp_path(image_bytes)
        warnings = []

        # LLM extraction (default or auto mode)
        if mode in ("llm", "auto") and not calibration_points:
            try:
                # Run LLM extraction in thread pool to avoid blocking
                llm_result, llm_conf = await asyncio.to_thread(
                    extract_chart_llm, temp_path
                )

                if "error" not in llm_result and llm_result.get("data"):
                    # LLM extraction succeeded
                    data = llm_result.get("data", [])
                    csv_content = llm_result_to_csv(llm_result)
                    chart_type_detected = llm_result.get("chart_type", "unknown")

                    processing_time = int((time.time() - start) * 1000)

                    return ExtractionResult(
                        success=True,
                        chart_type=chart_type_detected,
                        confidence=round(llm_conf, 3),
                        data=data,
                        csv=csv_content,
                        warnings=warnings,
                        processing_time_ms=processing_time
                    )
                elif mode == "llm":
                    # LLM mode only, but failed
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM extraction failed: {llm_result.get('error', 'No data extracted')}"
                    )
                else:
                    # Auto mode, fall back to CV
                    warnings.append("[LLM_FALLBACK] LLM extraction failed, using CV pipeline")

            except HTTPException:
                raise
            except Exception as e:
                if mode == "llm":
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM extraction error: {str(e)}"
                    )
                warnings.append(f"[LLM_FALLBACK] LLM error: {str(e)}")

        # CV extraction (fallback or explicit or calibrated)
        result = await asyncio.to_thread(
            extract_chart,
            image_path=temp_path,
            chart_type=ChartType(chart_type) if chart_type else None,
            x_scale=Scale(x_scale),
            y_scale=Scale(y_scale),
            calibration_points=calibration_points,
            use_mistral=use_mistral,
            generate_overlay_image=False
        )

        # Build CSV
        csv_lines = ["x,y"]
        for point in result.data:
            csv_lines.append(f"{point[0]},{point[1]}")
        csv_content = "\n".join(csv_lines)

        # Parse to data
        data = parse_csv_to_data(csv_content)

        # Collect warnings
        warnings.extend([f"[{w.code.value}] {w.message}" for w in result.warnings])
        if calibration_points:
            warnings.insert(0, "[CALIBRATED] Using user-provided calibration points")

        processing_time = int((time.time() - start) * 1000)

        return ExtractionResult(
            success=True,
            chart_type=result.chart_type.value,
            confidence=round(result.confidence.overall(), 3),
            data=data,
            csv=csv_content,
            warnings=warnings,
            processing_time_ms=processing_time
        )

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# --- Routes ---

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - start_time, 2)
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - start_time, 2)
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@v1_router.post("/extract", response_model=ExtractionResult)
async def extract_data_v1(
    file: UploadFile = File(..., description="Chart image (PNG, JPG, WebP)"),
    mode: str = "llm",
    chart_type: Optional[str] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    client_ip: str = Depends(get_client_ip)
):
    """
    Extract data from a chart image.

    **Extraction modes:**
    - `llm`: Use LLM vision (Pixtral) for direct extraction (default, recommended)
    - `cv`: Use computer vision pipeline with OCR
    - `auto`: Try LLM first, fall back to CV if it fails

    **Supported chart types:**
    - Line charts, Bar charts, Scatter plots

    **Not supported:**
    - Heatmaps, pie charts, treemaps, GitHub contribution graphs

    **Parameters:**
    - `file`: Chart image file (PNG, JPG, WebP)
    - `mode`: Extraction mode: llm (default), cv, auto
    - `chart_type`: Force chart type (scatter, line, bar). Auto-detected if not specified.

    **Returns:**
    - `data`: List of extracted data points
    - `csv`: CSV string
    - `confidence`: Extraction confidence (0-1)
    """
    return await _extract_data_impl(file, mode, chart_type, x_scale, y_scale, client_ip)


# Legacy endpoint with deprecation warning
@app.post("/extract", response_model=ExtractionResult, deprecated=True, tags=["Legacy"])
async def extract_data_legacy(
    file: UploadFile = File(..., description="Chart image (PNG, JPG, WebP)"),
    mode: str = "llm",
    chart_type: Optional[str] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    client_ip: str = Depends(get_client_ip)
):
    """[DEPRECATED] Use /v1/extract instead. This endpoint will be removed in 6 months."""
    logger.warning("Deprecated endpoint /extract called. Use /v1/extract instead.")
    return await _extract_data_impl(file, mode, chart_type, x_scale, y_scale, client_ip)


async def _extract_data_impl(
    file: UploadFile,
    mode: str,
    chart_type: Optional[str],
    x_scale: str,
    y_scale: str,
    client_ip: str
):
    """Shared implementation for extract endpoints."""
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per minute."
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (PNG, JPG, WebP)."
        )

    try:
        # Read image
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )

        # Process extraction using shared logic
        return await _process_chart_extraction(
            image_bytes=image_bytes,
            mode=mode,
            chart_type=chart_type,
            x_scale=x_scale,
            y_scale=y_scale,
            use_mistral=True
        )

    except HTTPException:
        raise
    except ValueError as e:
        # Image validation errors from image_to_temp_path
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@v1_router.post("/extract/base64", response_model=ExtractionResult)
async def extract_data_base64_v1(
    image_base64: str,
    mode: str = "llm",
    chart_type: Optional[str] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    use_mistral: bool = True,
    client_ip: str = Depends(get_client_ip)
):
    """
    Extract data from a base64-encoded chart image.

    Same as /v1/extract but accepts base64 string instead of file upload.

    **Parameters:**
    - `image_base64`: Base64-encoded image (with or without data URI prefix)
    - `mode`: Extraction mode (llm, cv, auto)
    - `chart_type`: Optional chart type override
    - `x_scale`, `y_scale`: Axis scales (linear or log)
    - `use_mistral`: Use Mistral OCR for CV mode

    **Returns:**
    - Same as /v1/extract endpoint
    """
    return await _extract_base64_impl(image_base64, mode, chart_type, x_scale, y_scale, use_mistral, client_ip)


# Legacy base64 endpoint
@app.post("/extract/base64", response_model=ExtractionResult, deprecated=True, tags=["Legacy"])
async def extract_data_base64_legacy(
    image_base64: str,
    mode: str = "llm",
    chart_type: Optional[str] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    use_mistral: bool = True,
    client_ip: str = Depends(get_client_ip)
):
    """[DEPRECATED] Use /v1/extract/base64 instead. This endpoint will be removed in 6 months."""
    logger.warning("Deprecated endpoint /extract/base64 called. Use /v1/extract/base64 instead.")
    return await _extract_base64_impl(image_base64, mode, chart_type, x_scale, y_scale, use_mistral, client_ip)


async def _extract_base64_impl(
    image_base64: str,
    mode: str,
    chart_type: Optional[str],
    x_scale: str,
    y_scale: str,
    use_mistral: bool,
    client_ip: str
):
    """Shared implementation for base64 extract endpoints."""
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per minute."
        )

    try:
        # Decode base64
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image too large. Maximum size is 10MB."
            )

        # Process extraction using shared logic
        return await _process_chart_extraction(
            image_bytes=image_bytes,
            mode=mode,
            chart_type=chart_type,
            x_scale=x_scale,
            y_scale=y_scale,
            use_mistral=use_mistral
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Base64 extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@v1_router.post("/extract/calibrated", response_model=ExtractionResult)
async def extract_calibrated_v1(
    file: UploadFile = File(..., description="Chart image"),
    calibration_json: str = None,
    client_ip: str = Depends(get_client_ip)
):
    """
    Extract data using user-provided calibration points.

    **Use this for Dense charts where automatic extraction fails.**

    The user provides reference points mapping pixel positions to actual values.
    The API then extracts data points and applies the calibration transform.

    **calibration_json format:**
    ```json
    {
        "x_axis": [
            {"pixel": 100, "value": 0},
            {"pixel": 500, "value": 20}
        ],
        "y_axis": [
            {"pixel": 350, "value": 0},
            {"pixel": 50, "value": 30}
        ]
    }
    ```

    Provide at least 2 points per axis for linear interpolation.
    """
    return await _extract_calibrated_impl(file, calibration_json, client_ip)


# Legacy calibrated endpoint
@app.post("/extract/calibrated", response_model=ExtractionResult, deprecated=True, tags=["Legacy"])
async def extract_calibrated_legacy(
    file: UploadFile = File(..., description="Chart image"),
    calibration_json: str = None,
    client_ip: str = Depends(get_client_ip)
):
    """[DEPRECATED] Use /v1/extract/calibrated instead. This endpoint will be removed in 6 months."""
    logger.warning("Deprecated endpoint /extract/calibrated called. Use /v1/extract/calibrated instead.")
    return await _extract_calibrated_impl(file, calibration_json, client_ip)


async def _extract_calibrated_impl(
    file: UploadFile,
    calibration_json: str,
    client_ip: str
):
    """Shared implementation for calibrated extract endpoints."""
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 20 requests per minute.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

        # Parse calibration JSON
        import json
        calibration = None
        if calibration_json:
            try:
                calibration = json.loads(calibration_json)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid calibration JSON: {str(e)}"
                )

        # Process extraction with calibration
        return await _process_chart_extraction(
            image_bytes=image_bytes,
            mode="cv",  # Calibration requires CV pipeline
            calibration_points=calibration,
            use_mistral=True
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Calibrated extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Calibrated extraction failed: {str(e)}"
        )


# Register v1 router
app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
