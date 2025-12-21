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

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

# CORS - Configure allowed origins from environment
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    # Default allowed origins for production
    ALLOWED_ORIGINS = [
        "https://kiku-jw.github.io",
        "https://chart2csv.kikuai.dev",
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


@app.post("/extract", response_model=ExtractionResult)
async def extract_data(
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
    
    start = time.time()
    
    try:
        # Read image
        image_bytes = await file.read()
        
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Save to temp file
        temp_path = image_to_temp_path(image_bytes)
        
        try:
            warnings = []
            
            # LLM extraction (default)
            if mode in ("llm", "auto"):
                try:
                    llm_result, llm_conf = extract_chart_llm(temp_path)
                    
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
                        
                except Exception as e:
                    if mode == "llm":
                        raise HTTPException(
                            status_code=500,
                            detail=f"LLM extraction error: {str(e)}"
                        )
                    warnings.append(f"[LLM_FALLBACK] LLM error: {str(e)}")
            
            # CV extraction (fallback or explicit)
            result = extract_chart(
                image_path=temp_path,
                chart_type=ChartType(chart_type) if chart_type else None,
                x_scale=Scale(x_scale),
                y_scale=Scale(y_scale),
                use_mistral=True,
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
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@app.post("/extract/base64", response_model=ExtractionResult)
async def extract_data_base64(
    image_base64: str,
    chart_type: Optional[str] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    use_mistral: bool = True,
    client_ip: str = Depends(get_client_ip)
):
    """
    Extract data from a base64-encoded chart image.
    
    Same as /extract but accepts base64 string instead of file upload.
    """
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per minute."
        )
    
    start = time.time()
    
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
        
        # Save to temp file
        temp_path = image_to_temp_path(image_bytes)
        
        try:
            result = extract_chart(
                image_path=temp_path,
                chart_type=ChartType(chart_type) if chart_type else None,
                x_scale=Scale(x_scale),
                y_scale=Scale(y_scale),
                use_mistral=use_mistral,
                generate_overlay_image=False
            )
            
            csv_lines = ["x,y"]
            for point in result.data:
                csv_lines.append(f"{point[0]},{point[1]}")
            csv_content = "\n".join(csv_lines)
            
            data = parse_csv_to_data(csv_content)
            warnings = [f"[{w.code.value}] {w.message}" for w in result.warnings]
            
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
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@app.post("/extract/calibrated", response_model=ExtractionResult)
async def extract_calibrated(
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
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    start = time.time()
    
    try:
        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        temp_path = image_to_temp_path(image_bytes)
        
        try:
            import json
            calibration = None
            if calibration_json:
                try:
                    calibration = json.loads(calibration_json)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid calibration JSON")
            
            # Use CV pipeline with calibration points
            result = extract_chart(
                image_path=temp_path,
                calibration_points=calibration,
                use_mistral=True,
                generate_overlay_image=False
            )
            
            csv_lines = ["x,y"]
            for point in result.data:
                csv_lines.append(f"{point[0]},{point[1]}")
            csv_content = "\n".join(csv_lines)
            
            data = parse_csv_to_data(csv_content)
            warnings = [f"[{w.code.value}] {w.message}" for w in result.warnings]
            if calibration:
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
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibrated extraction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
