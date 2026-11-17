import hashlib
from dataclasses import dataclass

@dataclass(frozen=True)
class AdapterManifest:
    base_model: str
    quant_bits: int
    tokenizer_sha256: str

def verify_tokenizer(tokenizer_bytes: bytes, manifest: AdapterManifest) -> bool:
    digest = hashlib.sha256(tokenizer_bytes).hexdigest()
    return digest == manifest.tokenizer_sha256
