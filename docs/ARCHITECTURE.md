# Architecture: QLoRA Adapter Price Service

## Overview

`qlora-adapter-price-service` wraps a 4-bit NF4 quantized LoRA price predictor with strict manifest verification, out-of-distribution (OOD) gating, and runtime telemetry.

## Components

1. **Manifest & Tokenizer Verifier (`service/`)**
   - Validates `AdapterManifest` metadata (`base_model`, `adapter_revision`, `quantization_bits == 4`).
   - Computes and compares `tokenizer_sha256` before loading adapter weights to prevent silent vocabulary drift.

2. **OOD & Confidence Gate (`src/`)**
   - Evaluates incoming product descriptions against supported categories and token-length bounds.
   - Routes OOD or malformed requests to a deterministic fallback pricer and flags `used_fallback=True`.

3. **Telemetry Collector**
   - Captures peak VRAM usage (MB), p50/p95 inference latency (ms), and evaluation MAE.

## Failure Modes & Mitigations

| Failure Mode | Mitigation |
| --- | --- |
| Tokenizer mismatch across checkpoints | Abort load on SHA-256 mismatch |
| Out-of-distribution product category | Deterministic fallback pricer |
| Unparseable / negative model output | Regex extraction + range clamping |
