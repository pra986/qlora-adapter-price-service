from __future__ import annotations

import json
from src.engine import QLoRAPriceEngine
from src.models import AdapterManifest, PriceInferenceRequest

DEMO_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def main() -> None:
    manifest = AdapterManifest(
        base_model_id="meta-llama/Meta-Llama-3.1-8B",
        adapter_revision="v1.0.0",
        quantization_bits=4,
        tokenizer_sha256=DEMO_HASH,
        supported_categories=["Electronics", "Appliances", "Home"],
        vram_budget_mb=12288,
    )
    engine = QLoRAPriceEngine(manifest)
    req = PriceInferenceRequest(
        request_id="demo-001",
        item_title="Sony WH-1000XM5 Wireless Headphones",
        category="Electronics",
        description="Noise-canceling over-ear Bluetooth headphones with 30h battery",
        tokenizer_sha256=DEMO_HASH,
    )
    res = engine.predict(req)
    print(json.dumps(res.model_dump(), indent=2))


if __name__ == "__main__":
    main()
