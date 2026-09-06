"""Validated, name-based tensor file contract for the Rin QNN JNI pipeline."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np

MANIFEST_NAME = ".rin_qnn_outputs.json"


def _checked_record(root: Path, item: dict) -> tuple[Path, dict]:
    name = str(item.get("file", ""))
    if not name or name in (".", "..") or any(c in name for c in ("/", "\\", ":")):
        raise ValueError(f"Unsafe tensor output filename: {name!r}")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"QNN output missing: tensor={item.get('name')} file={path}")
    shape = item.get("shape")
    if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(f"Invalid QNN output shape: {item}")
    size = path.stat().st_size
    expected = int(item.get("bytes", 0))
    if expected <= 0 or size != expected:
        raise ValueError(f"QNN output size mismatch: tensor={item.get('name')} expected={expected} actual={size} file={path}")
    dtype = item.get("dtype")
    if dtype in ("float32", "float16") and expected != math.prod(shape) * (4 if dtype == "float32" else 2):
        raise ValueError(f"QNN output dtype/shape/bytes disagree: {item}")
    return path, item


def output_records(result_dir) -> list[dict] | None:
    root = Path(result_dir)
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("complete") is not True:
        raise ValueError(f"Incomplete or unsupported QNN output manifest: {manifest}")
    records = data.get("outputs")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Empty QNN output manifest: {manifest}")
    files = [x.get("file") for x in records]
    keys = [(x.get("name"), x.get("dtype")) for x in records]
    if len(set(files)) != len(files) or len(set(keys)) != len(keys):
        raise ValueError(f"Ambiguous QNN output manifest: {manifest}")
    return records


def resolve_output(result_dir, tensor_name: str, legacy_index: int | None = None,
                   expected_elements: int | None = None, allow_native: bool = False) -> tuple[Path, dict]:
    """Prefer semantic names; numbered legacy aliases never use directory sorting."""
    root = Path(result_dir)
    names = [tensor_name]
    if legacy_index is not None:
        names += [f"output_{legacy_index}", f"Output_{legacy_index}"]
    records = output_records(root)
    if records is not None:
        for name in names:
            matches = [x for x in records if x.get("name") == name]
            if not matches:
                continue
            floats = [x for x in matches if x.get("dtype") == "float32"]
            eligible = floats or ([x for x in matches if x.get("dtype") == "float16"] if allow_native else [])
            if len(eligible) != 1:
                raise ValueError(f"QNN output dtype/alias conflict for {tensor_name}: {matches}")
            path, item = _checked_record(root, eligible[0])
            if expected_elements is not None and math.prod(item["shape"]) != expected_elements:
                raise ValueError(f"QNN output element count mismatch: tensor={tensor_name} expected={expected_elements} shape={item['shape']}")
            return path, item
        raise FileNotFoundError(f"QNN output tensor={tensor_name} aliases={names} not found; actual={[x.get('name') for x in records]} dir={root}")
    for native in ([False, True] if allow_native else [False]):
        for name in names:
            path = root / (name + ("_native.raw" if native else ".raw"))
            if not path.is_file():
                continue
            size = path.stat().st_size
            dtype = "float32"
            if expected_elements is not None:
                if size == expected_elements * 4:
                    dtype = "float32"
                elif allow_native and size == expected_elements * 2:
                    dtype = "float16"
                else:
                    raise ValueError(f"QNN raw size mismatch: tensor={tensor_name} bytes={size} elements={expected_elements} file={path}")
            elif size <= 0 or size % 4 or native:
                raise ValueError(f"Invalid/unknown raw precision: {path} bytes={size}")
            if size <= 0:
                raise ValueError(f"Empty QNN output: {path}")
            return path, {"name": name, "file": path.name, "bytes": size, "dtype": dtype, "shape": None}
    available = sorted(p.name for p in root.glob("*.raw"))[:32]
    raise FileNotFoundError(f"QNN output tensor={tensor_name} aliases={names} missing; actual={available} dir={root}")


def read_float_output(result_dir, tensor_name, shape, legacy_index=None, allow_native=False):
    shape = tuple(shape)
    path, info = resolve_output(result_dir, tensor_name, legacy_index, math.prod(shape), allow_native)
    data = np.fromfile(path, dtype=np.float32 if info["dtype"] == "float32" else np.float16).astype(np.float32, copy=False)
    if data.size != math.prod(shape) or not np.isfinite(data).all():
        raise ValueError(f"Invalid QNN output data: tensor={tensor_name} file={path} size={data.size}; expected={math.prod(shape)} finite={bool(np.isfinite(data).all())}")
    actual = tuple(info["shape"]) if info.get("shape") else shape
    array = data.reshape(actual)
    if actual == shape:
        return array
    if len(actual) == len(shape) == 4:
        if tuple(actual[i] for i in (0, 3, 1, 2)) == shape:
            return np.ascontiguousarray(array.transpose(0, 3, 1, 2))
        if tuple(actual[i] for i in (0, 2, 3, 1)) == shape:
            return np.ascontiguousarray(array.transpose(0, 2, 3, 1))
    raise ValueError(f"Unsupported QNN layout: tensor={tensor_name} actual={actual} expected={shape}")


def named_input(name, path):
    name, path = str(name), str(path)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or any(c.isspace() for c in path):
        raise ValueError(f"Invalid QNN input-list entry: {name!r} {path!r}")
    return f"{name}:={path}"


def write_input_rows(path, rows):
    """Rewrite changed lists after upgrades; retain identical lists without I/O churn."""
    rows = [list(row) for row in rows]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("QNN input-list rows must be non-empty with equal column counts")
    first_names = None
    for row in rows:
        if any(not isinstance(x, str) or not x or any(c.isspace() for c in x) for x in row):
            raise ValueError("QNN input-list contains blank or whitespace-separated file paths")
        named = [":=" in x for x in row]
        if any(named) and not all(named):
            raise ValueError("QNN input-list cannot mix named and positional entries")
        names = [x.split(":=", 1)[0] for x in row] if all(named) else []
        if names and len(set(names)) != len(names):
            raise ValueError("Duplicate QNN input name")
        if first_names is not None and names != first_names:
            raise ValueError("QNN batched rows have inconsistent input-name order")
        first_names = names
    text = "".join(" ".join(r) + "\n" for r in rows)
    target = Path(path)
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".rin-next")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, target)


def validate_input_list(path, native_input=False):
    rows = [l.split() for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith(("#", "%"))]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError(f"QNN input_preflight: invalid row/column count in {path}")
    for row_idx, row in enumerate(rows):
        for col, entry in enumerate(row):
            name, _, raw = entry.partition(":=")
            raw = raw if _ else entry
            file = Path(raw)
            size = file.stat().st_size if file.is_file() else -1
            if size <= 0 or (not native_input and size % 4):
                raise ValueError(f"QNN input_preflight: row={row_idx} column={col} tensor={name if _ else '?'} bytes={size} file={raw}")
    return len(rows)


def prepare_output_dir(output_dir, work_dir):
    root, out = Path(work_dir).resolve(), Path(output_dir).resolve()
    if root == out or root not in out.parents:
        raise ValueError(f"Refusing to clear outputs outside the generation work directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if re.fullmatch(r"Result_\d+", p.name):
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)


def validate_output_tree(output_dir, expected_results):
    """A JNI success must provide fresh, complete manifests for every input row."""
    count = 0
    for idx in range(expected_results):
        root = Path(output_dir) / f"Result_{idx}"
        records = output_records(root)
        if records is None:
            raise FileNotFoundError(f"QNN output_validate: result={idx} manifest missing in {root}")
        for item in records:
            _checked_record(root, item)
            count += 1
    return count
