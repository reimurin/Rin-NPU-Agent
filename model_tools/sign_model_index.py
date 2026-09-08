#!/usr/bin/env python3
"""Sign a Rin NPU model-pack-index.json with the local Ed25519 release key.

The output is a detached Base64 Ed25519 signature over the index's exact UTF-8 bytes.
Do not rewrite the index after signing. The private key remains outside the repository.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from sign_model_manifest import DEFAULT_KEY_ID, find_default_private_key, load_private_key


def sign_index(index: Path, private_key: Path, output: Path) -> str:
    raw = index.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("schema") != 1:
        raise ValueError("model-pack index must declare schema=1")
    packages = parsed.get("packages")
    if not isinstance(packages, list):
        raise ValueError("model-pack index must contain a packages array")

    key = load_private_key(private_key)
    signature = key.sign(raw)
    key.public_key().verify(signature, raw)
    encoded = base64.b64encode(signature).decode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(encoded + "\n", encoding="ascii", newline="\n")
    tmp.replace(output)
    return encoded


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign Rin NPU model-pack index with Ed25519")
    parser.add_argument("index", type=Path, help="Path to model-pack-index.json")
    parser.add_argument("--key", type=Path, help="Private Ed25519 PEM; defaults to the local release key")
    parser.add_argument("--output", type=Path, help="Detached Base64 signature; default: <index>.sig")
    args = parser.parse_args()

    index = args.index.expanduser().resolve()
    if not index.is_file():
        raise FileNotFoundError(index)
    key = args.key.expanduser().resolve() if args.key else find_default_private_key()
    if key is None or not key.is_file():
        raise FileNotFoundError("Ed25519 private key not found; pass --key or set RIN_MODEL_PACK_SIGNING_KEY")
    output = args.output.expanduser().resolve() if args.output else index.with_name(index.name + ".sig")
    sign_index(index, key, output)
    print(f"SIGNED={index}")
    print(f"SIGNATURE={output}")
    print(f"KEY_ID={DEFAULT_KEY_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
