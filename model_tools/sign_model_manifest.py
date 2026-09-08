#!/usr/bin/env python3
"""Sign a Rin NPU model_manifest.json with the local Ed25519 release key.

The manifest is signed byte-for-byte. Do not rewrite or pretty-print the manifest after
this script runs; any byte change intentionally invalidates the detached signature.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

DEFAULT_KEY_ID = "rin-model-pack-ed25519-2026-09-01"
PRIVATE_KEY_NAME = "rin_model_pack_ed25519_private.pem"


def find_default_private_key() -> Optional[Path]:
    env = os.environ.get("RIN_MODEL_PACK_SIGNING_KEY", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    project_root = here.parent.parent
    candidate = project_root.parent.parent / "toolchain" / "model-pack-signing" / "private" / PRIVATE_KEY_NAME
    if candidate.is_file():
        return candidate.resolve()
    candidate = project_root / "toolchain" / "model-pack-signing" / "private" / PRIVATE_KEY_NAME
    if candidate.is_file():
        return candidate.resolve()
    return None


def load_private_key(path: Path) -> Ed25519PrivateKey:
    loaded = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(loaded, Ed25519PrivateKey):
        raise TypeError(f"Signing key is not Ed25519: {path}")
    return loaded


def sign_manifest(manifest: Path, private_key: Path, output: Path, key_id: str) -> dict:
    raw = manifest.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("model manifest must be a JSON object")
    if parsed.get("schema") != 1 or parsed.get("complete") is not True:
        raise ValueError("model manifest must declare schema=1 and complete=true")

    key = load_private_key(private_key)
    signature = key.sign(raw)
    key.public_key().verify(signature, raw)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    document = {
        "schema": 1,
        "algorithm": "Ed25519",
        "key_id": key_id,
        "manifest_sha256": manifest_sha,
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(output)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign Rin NPU model manifest with Ed25519")
    parser.add_argument("manifest", type=Path, help="Path to model_manifest.json")
    parser.add_argument("--key", type=Path, help="Private Ed25519 PEM. Defaults to RIN_MODEL_PACK_SIGNING_KEY or local toolchain key.")
    parser.add_argument("--key-id", default=DEFAULT_KEY_ID)
    parser.add_argument("--output", type=Path, help="Detached signature JSON path; default: model_manifest.sig.json next to manifest")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    key = args.key.expanduser().resolve() if args.key else find_default_private_key()
    if key is None or not key.is_file():
        raise FileNotFoundError("Ed25519 private key not found; pass --key or set RIN_MODEL_PACK_SIGNING_KEY")
    output = args.output.expanduser().resolve() if args.output else manifest.with_name("model_manifest.sig.json")
    result = sign_manifest(manifest, key, output, args.key_id)
    print(f"SIGNED={manifest}")
    print(f"SIGNATURE={output}")
    print(f"KEY_ID={result['key_id']}")
    print(f"MANIFEST_SHA256={result['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
