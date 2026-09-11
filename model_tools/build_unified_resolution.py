from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_resolution(raw: str) -> tuple[int, int]:
    try:
        w_text, h_text = raw.lower().split("x", 1)
        width, height = int(w_text), int(h_text)
    except Exception as exc:
        raise SystemExit(f"Invalid resolution {raw!r}; expected WxH") from exc
    if width not in range(64, 8193) or height not in range(64, 8193) or width % 8 or height % 8:
        raise SystemExit(f"Unsupported resolution {width}x{height}; dimensions must be 64..8192 and divisible by 8")
    return width, height


def model_id_for(width: int, height: int) -> str:
    if (width, height) == (1024, 1024):
        return "wai-v170-sm8750-lora-r64-1024-partitioned-v1"
    return f"wai-v170-sm8750-lora-r64-{width}x{height}-partitioned-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(pending, path)


def state(root: Path, stage: str, **extra: object) -> None:
    payload = {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    atomic_json(root / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_logged(command: list[str], log_file: Path, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write("CMD=" + subprocess.list2cmdline(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"Command failed rc={result.returncode}: {command[0]}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_split_unet(merged_unet: Path, upstream_root: Path, out_dir: Path, width: int, height: int) -> None:
    expected = [out_dir / "encoder/model.onnx", out_dir / "encoder/model.data", out_dir / "decoder/model.onnx", out_dir / "decoder/model.data"]
    if all(item.is_file() and item.stat().st_size > 0 for item in expected):
        return
    if any(item.exists() for item in expected):
        raise RuntimeError("Partial split UNet export exists; inspect instead of overwriting")
    exporter = upstream_root / "SDXL/debug/export_split_unet.py"
    if not exporter.is_file():
        raise RuntimeError(f"Split exporter missing: {exporter}")
    if not (merged_unet / "config.json").is_file() or not list(merged_unet.glob("*.safetensors")):
        raise RuntimeError(f"Merged UNet is incomplete: {merged_unet}")
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    original_export = torch.onnx.export

    def legacy_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return original_export(*args, **kwargs)

    torch.onnx.export = legacy_export
    try:
        module = load_module(exporter, "rin_alpha5_split_export")
        module.MERGED_UNET_DIR = merged_unet
        module.ONNX_DIR = out_dir
        module.ONNX_ENC = out_dir / "encoder"
        module.ONNX_DEC = out_dir / "decoder"
        module.export_onnx(width=width, height=height)
    finally:
        torch.onnx.export = original_export
    missing = [item for item in expected if not item.is_file() or item.stat().st_size <= 0]
    if missing:
        raise RuntimeError("Split UNet export incomplete: " + ", ".join(map(str, missing)))


def export_vae(diffusers_dir: Path, upstream_root: Path, out_dir: Path, width: int, height: int) -> Path:
    onnx_path = out_dir / "vae_decoder.onnx"
    if onnx_path.is_file() and onnx_path.stat().st_size > 0:
        return onnx_path
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError("Partial VAE ONNX export exists; inspect instead of overwriting")
    out_dir.mkdir(parents=True, exist_ok=True)
    script = upstream_root / "SDXL/export_sdxl_to_onnx.py"
    command = [
        sys.executable, str(script), "--diffusers-dir", str(diffusers_dir), "--out-dir", str(out_dir),
        "--component", "vae", "--resolution", f"{width}x{height}", "--opset", "17",
        "--onnx-exporter", "legacy", "--skip-validate",
    ]
    run_logged(command, out_dir / "export.log", cwd=upstream_root, timeout=4 * 3600)
    if not onnx_path.is_file() or onnx_path.stat().st_size <= 0:
        raise RuntimeError("VAE ONNX export did not produce vae_decoder.onnx")
    return onnx_path


def validate_vae_onnx(path: Path, width: int, height: int) -> None:
    import onnx
    model = onnx.load(str(path), load_external_data=False)

    def dims(value):
        return [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim]

    inputs = {item.name: dims(item) for item in model.graph.input}
    outputs = {item.name: dims(item) for item in model.graph.output}
    expected_input = [1, 4, height // 8, width // 8]
    expected_output = [1, 3, height, width]
    if inputs.get("latent") != expected_input or outputs.get("image") != expected_output:
        raise RuntimeError(f"VAE ONNX shape mismatch: inputs={inputs}, outputs={outputs}, expected={expected_input}->{expected_output}")


def compile_vae(onnx_path: Path, sdk_root: Path, out_path: Path, tmp_dir: Path) -> None:
    if out_path.is_file() and out_path.stat().st_size > 0:
        return
    if out_path.exists():
        raise RuntimeError(f"Invalid existing VAE context: {out_path}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(QAIRT_SDK_ROOT=str(sdk_root), QNN_SDK_ROOT=str(sdk_root), QAIRT_TMP_DIR=str(tmp_dir), TMP=str(tmp_dir), TEMP=str(tmp_dir))
    os.environ["PATH"] = str(sdk_root / "lib/x86_64-windows-msvc") + os.pathsep + os.environ.get("PATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    import qairt
    model = qairt.convert(str(onnx_path), float_precision=16)
    compiled = qairt.compile(model, config=qairt.CompileConfig(backend="HTP", soc_details="chipset:SM8750", log_level="info"))
    saved = Path(compiled.save(str(out_path)))
    if not saved.is_file():
        saved = out_path
    if not saved.is_file() or saved.stat().st_size <= 0:
        raise RuntimeError("VAE context was not saved")


def context_graph(sdk_root: Path, context: Path, report: Path) -> dict:
    utility = sdk_root / "bin/x86_64-windows-msvc/qnn-context-binary-utility.exe"
    if not utility.is_file():
        raise RuntimeError(f"QNN context utility missing: {utility}")
    env = os.environ.copy()
    env["PATH"] = str(sdk_root / "lib/x86_64-windows-msvc") + os.pathsep + env.get("PATH", "")
    run_logged([str(utility), f"--context_binary={context}", f"--json_file={report}"], report.with_suffix(".log"), env=env, timeout=120)
    data = json.loads(report.read_text(encoding="utf-8"))
    graphs = data.get("info", {}).get("graphs", [])
    if len(graphs) != 1:
        raise RuntimeError("Expected exactly one graph in compiled VAE context")
    return graphs[0]["info"]


def validate_vae_context(sdk_root: Path, context: Path, report: Path, width: int, height: int) -> dict:
    graph = context_graph(sdk_root, context, report)
    inputs = {item["info"]["name"]: item["info"]["dimensions"] for item in graph["graphInputs"]}
    outputs = {item["info"]["name"]: item["info"]["dimensions"] for item in graph["graphOutputs"]}
    expected_input = [1, 4, height // 8, width // 8]
    expected_output = [1, 3, height, width]
    if inputs.get("latent") != expected_input or outputs.get("image") != expected_output:
        raise RuntimeError(f"Compiled VAE layout mismatch: {inputs} -> {outputs}")
    return {"graph": graph.get("graphName"), "inputs": inputs, "outputs": outputs}


def finalize_seven_manifest(job_root: Path, width: int, height: int) -> dict:
    package = job_root / "package"
    manifest_path = package / "lora_template.json"
    if not manifest_path.is_file():
        raise RuntimeError("Seven-part compiler did not produce lora_template.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("rank_capacity") != 64:
        raise RuntimeError("Seven-part manifest is incomplete or wrong rank capacity")
    parts = [part for stage in ("encoder", "decoder") for part in manifest.get("graphs", {}).get(stage, {}).get("parts", [])]
    if len(parts) != 7:
        raise RuntimeError(f"Expected seven UNet parts, got {len(parts)}")
    first_inputs = manifest["graphs"]["encoder"]["parts"][0].get("inputs", [])
    sample = next((item for item in first_inputs if item.get("name") == "sample"), None)
    expected_sample = [1, 4, height // 8, width // 8]
    if sample is None or sample.get("shape") != expected_sample:
        raise RuntimeError(f"Seven-part source resolution mismatch: sample={sample}, expected={expected_sample}")
    manifest["model_id"] = model_id_for(width, height)
    manifest["resolution"] = [width, height]
    atomic_json(manifest_path, manifest)
    for item in parts:
        file = package / item["context_file"]
        if not file.is_file() or file.stat().st_size != item["context_bytes"] or sha256(file) != item["context_sha256"]:
            raise RuntimeError(f"Seven-part file integrity mismatch: {item['id']}")
    return manifest


def compile_seven(source: Path, app_root: Path, sdk_root: Path, job_root: Path, width: int, height: int) -> dict:
    status = job_root / "status.json"
    if job_root.exists():
        if status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("stage") == "COMPLETE_OFFLINE":
            return json.loads((job_root / "package/lora_template.json").read_text(encoding="utf-8"))
        raise RuntimeError(f"Existing incomplete seven-part job; inspect instead of resubmitting: {job_root}")
    compiler = app_root / "model_tools/partitioned/compile_partitioned_contexts.py"
    command = [sys.executable, "-B", str(compiler), "--source", str(source), "--sdk", str(sdk_root), "--out", str(job_root), "--capacity", "64", "--resolution", f"{width}x{height}"]
    run_logged(command, job_root.parent / "seven_compile.log", cwd=compiler.parent, timeout=24 * 3600)
    return json.loads((job_root / "package/lora_template.json").read_text(encoding="utf-8"))


def deploy_manifest(root: Path, seven_root: Path, vae_context: Path, width: int, height: int, manifest: dict, vae_layout: dict) -> dict:
    model_id = model_id_for(width, height)
    files = []
    package = seven_root / "package"
    for name in ["lora_template.json"] + [item["context_file"] for stage in ("encoder", "decoder") for item in manifest["graphs"][stage]["parts"]]:
        source = package / name
        files.append({"source": str(source), "phone_relative": f"context/lora/{model_id}/{name}", "bytes": source.stat().st_size, "sha256": sha256(source)})
    files.append({"source": str(vae_context), "phone_relative": f"context/{width}x{height}/vae_decoder.serialized.bin.bin", "bytes": vae_context.stat().st_size, "sha256": sha256(vae_context)})
    payload = {
        "schema": 1,
        "model_id": model_id,
        "resolution": [width, height],
        "soc": 69,
        "dsp": 79,
        "rank_capacity": 64,
        "files": files,
        "vae_layout": vae_layout,
        "phone_validated": False,
    }
    atomic_json(root / "deploy_manifest.json", payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build one resolution-specific unified WAI seven-part UNet + VAE component for SM8750")
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--merged-unet-dir", required=True)
    ap.add_argument("--diffusers-dir", required=True)
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--app-root", required=True)
    ap.add_argument("--sdk-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    width, height = parse_resolution(args.resolution)
    model_id = model_id_for(width, height)
    root = Path(args.out).resolve()
    merged_unet = Path(args.merged_unet_dir).resolve()
    diffusers_dir = Path(args.diffusers_dir).resolve()
    upstream_root = Path(args.upstream_root).resolve()
    app_root = Path(args.app_root).resolve()
    sdk_root = Path(args.sdk_root).resolve()
    plan = {
        "resolution": [width, height],
        "latent_nchw": [1, 4, height // 8, width // 8],
        "image_nchw": [1, 3, height, width],
        "model_id": model_id,
        "out": str(root),
        "merged_unet_dir": str(merged_unet),
        "diffusers_dir": str(diffusers_dir),
        "upstream_root": str(upstream_root),
        "app_root": str(app_root),
        "sdk_root": str(sdk_root),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return 0
    if (root / "result.json").is_file():
        raise SystemExit("Job already complete; inspect result.json instead of resubmitting")
    for required in (merged_unet, diffusers_dir, upstream_root / "SDXL", app_root / "model_tools", sdk_root):
        if not required.exists():
            raise SystemExit(f"Required input missing: {required}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        state(root, "EXPORT_UNET", model_id=model_id)
        split_dir = root / "onnx/unet_split"
        export_split_unet(merged_unet, upstream_root, split_dir, width, height)

        state(root, "EXPORT_VAE")
        vae_onnx = export_vae(diffusers_dir, upstream_root, root / "onnx/vae", width, height)
        validate_vae_onnx(vae_onnx, width, height)

        state(root, "COMPILE_SEVEN")
        seven_root = root / "seven"
        compile_seven(split_dir, app_root, sdk_root, seven_root, width, height)
        manifest = finalize_seven_manifest(seven_root, width, height)

        state(root, "COMPILE_VAE")
        vae_context = root / "contexts/vae_decoder_sm8750.bin"
        compile_vae(vae_onnx, sdk_root, vae_context, root / "tmp/vae")
        vae_layout = validate_vae_context(sdk_root, vae_context, root / "logs/vae_context.json", width, height)

        state(root, "FINALIZE")
        deployment = deploy_manifest(root, seven_root, vae_context, width, height, manifest, vae_layout)
        result = {**plan, "complete": True, "phone_validated": False, "deploy_manifest": str(root / "deploy_manifest.json"), "total_deploy_bytes": sum(item["bytes"] for item in deployment["files"])}
        atomic_json(root / "result.json", result)
        state(root, "COMPLETE_OFFLINE", total_deploy_bytes=result["total_deploy_bytes"], phone_validated=False)
        return 0
    except Exception as exc:
        state(root, "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
