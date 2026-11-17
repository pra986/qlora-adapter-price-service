# QLoRA Adapter Price Service (K7)

4-bit quantized LoRA open-model price predictor featuring an adapter manifest with tokenizer hash verification, out-of-distribution (OOD) fallback, and MAE/VRAM/latency telemetry.

## Architecture
- **4-bit QLoRA Manifest**: Verifies base model ID, adapter revision, and SHA-256 tokenizer hash before loading.
- **OOD Fallback**: Routes out-of-distribution or low-confidence inputs to a deterministic baseline pricer.
- **Performance Telemetry**: Tracks MAE, peak VRAM (MB), and p50/p95 inference latency.
