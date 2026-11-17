from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class AdapterManifest(BaseModel):
    base_model_id: str
    adapter_revision: str
    quantization_bits: Literal[4, 8] = 4
    lora_r: int = Field(gt=0, default=32)
    lora_alpha: int = Field(gt=0, default=64)
    tokenizer_sha256: str = Field(min_length=64, max_length=64)
    supported_categories: list[str] = Field(default_factory=list)
    vram_budget_mb: int = Field(gt=0, default=12288)


class PriceInferenceRequest(BaseModel):
    request_id: str = Field(min_length=1)
    item_title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tokenizer_sha256: str = Field(min_length=64, max_length=64)


class PriceInferenceResult(BaseModel):
    request_id: str
    predicted_price_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    used_fallback: bool = False
    fallback_reason: str | None = None
    latency_ms: float = Field(ge=0.0)
    vram_mb: int = Field(ge=0)
    mae_estimate_usd: float = Field(ge=0.0)
    status: Literal["ok", "fallback", "rejected"]
