# QLoRA Adapter Price Service (Project 07)

Manifest-verified QLoRA 4-bit price prediction service with tokenizer SHA-256 checks, out-of-distribution (OOD) fallback, and VRAM/latency telemetry.

## System Architecture

```
[Product Text / Metadata]
        |
        v
[1. Manifest & Tokenizer Verifier] ---> (SHA-256 / Base Model Check)
        |
        v
[2. OOD & Confidence Gate] -----------> (Fallback to Baseline Pricer if OOD)
        |
        v
[3. 4-Bit NF4 QLoRA Inference Engine] -> (VRAM & p50/p95 Latency Telemetry)
        |
        v
[4. Structured Price & Audit Output]
```

## 6-Stage Execution Workflow

1. **Manifest Validation**: Load `AdapterManifest` and verify base model ID, adapter revision, quantization config (`nf4`, double quant), and `tokenizer_sha256`.
2. **Input Normalization**: Sanitize product title, category, brand, and description; enforce token length bounds.
3. **OOD & Domain Gate**: Score input against supported product-category vocabulary and token-length distribution; route OOD or low-confidence inputs to a deterministic fallback pricer.
4. **4-Bit QLoRA Forward Pass**: Execute quantized inference with bounded `max_new_tokens` and deterministic decoding (`temperature=0.0`).
5. **Post-Processing & Bounds Check**: Parse numeric price prediction, clamp negative/implausible values, and attach confidence and fallback flags.
6. **Telemetry & Audit Logging**: Record MAE against reference labels, peak VRAM (MB), and p50/p95 inference latency (ms).

## Data & State Contract

- **AdapterManifest**: `model_id`, `adapter_revision`, `quantization_bits` (`4`), `tokenizer_sha256`, `max_seq_len`.
- **PriceRequest**: `item_id`, `title`, `category`, `description`, `reference_price` (optional).
- **PriceResponse**: `item_id`, `predicted_price`, `used_fallback`, `fallback_reason`, `vram_mb`, `latency_ms`, `manifest_verified`.

## Production Controls Added Over Course Baseline

- **Tokenizer SHA-256 Integrity Check**: Prevents silent embedding/vocabulary mismatch when loading LoRA adapters across checkpoints.
- **Deterministic OOD Fallback**: Protects downstream callers from hallucinated prices on unseen categories or malformed inputs.
- **VRAM & Latency Telemetry**: Tracks peak GPU memory and p50/p95 latency per batch for SLA enforcement.
- **Strict Output Parsing**: Rejects unparseable or out-of-range model outputs instead of propagating NaN/corrupt prices.

## Local Setup & Pytest Verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Interview Walkthrough

- **Why 4-bit QLoRA?** Cuts inference memory footprint by ~4x while preserving fine-tuned pricing accuracy on consumer GPUs.
- **Why verify tokenizer SHA-256?** A mismatched tokenizer shifts token IDs silently, degrading predictions without raising runtime errors.
- **How are OOD items handled?** Requests failing category or confidence thresholds bypass the adapter and return a deterministic baseline estimate with `used_fallback=True`.

## Pinned Ed Donner Commit Attribution

- Derived from and hardened beyond the [ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering) **Week 7** (fine-tuning open-source models with QLoRA) and **Week 8** (price prediction service integration) notebooks.

## Architecture
- **4-bit QLoRA Manifest**: Verifies base model ID, adapter revision, and SHA-256 tokenizer hash before loading.
- **OOD Fallback**: Routes out-of-distribution or low-confidence inputs to a deterministic baseline pricer.
- **Performance Telemetry**: Tracks MAE, peak VRAM (MB), and p50/p95 inference latency.
