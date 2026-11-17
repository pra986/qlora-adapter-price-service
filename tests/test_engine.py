import pytest
from src.engine import QLoRAPriceEngine, TokenizerMismatchError, VramBudgetExceededError
from src.models import AdapterManifest, PriceInferenceRequest

VALID_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
BAD_HASH = "0" * 64


def make_manifest(vram_budget_mb: int = 12288) -> AdapterManifest:
    return AdapterManifest(
        base_model_id="meta-llama/Meta-Llama-3.1-8B",
        adapter_revision="v1.0.0",
        quantization_bits=4,
        tokenizer_sha256=VALID_HASH,
        supported_categories=["Electronics", "Appliances"],
        vram_budget_mb=vram_budget_mb,
    )


def test_valid_prediction_and_idempotency():
    engine = QLoRAPriceEngine(make_manifest())
    req = PriceInferenceRequest(
        request_id="req-1",
        item_title="Sony Headphones",
        category="Electronics",
        description="Wireless noise canceling",
        tokenizer_sha256=VALID_HASH,
    )
    first = engine.predict(req)
    second = engine.predict(req)
    assert first.status == "ok"
    assert first.used_fallback is False
    assert first == second


def test_tokenizer_mismatch_raises():
    engine = QLoRAPriceEngine(make_manifest())
    req = PriceInferenceRequest(
        request_id="req-2",
        item_title="Blender",
        category="Appliances",
        description="High speed blender",
        tokenizer_sha256=BAD_HASH,
    )
    with pytest.raises(TokenizerMismatchError):
        engine.predict(req)


def test_ood_category_routes_to_fallback():
    engine = QLoRAPriceEngine(make_manifest())
    req = PriceInferenceRequest(
        request_id="req-3",
        item_title="Vintage Guitar",
        category="Musical Instruments",
        description="Acoustic guitar",
        tokenizer_sha256=VALID_HASH,
    )
    res = engine.predict(req)
    assert res.status == "fallback"
    assert res.used_fallback is True


def test_vram_budget_exceeded_raises():
    with pytest.raises(VramBudgetExceededError):
        QLoRAPriceEngine(make_manifest(vram_budget_mb=2048))
