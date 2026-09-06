"""Create fixed-rank, packed dynamic LoRA inputs without changing base weights.

The tool loads ONNX metadata only. External weights are linked into a new directory;
no original ONNX, context, or weight file is overwritten. Initial scope is ordinary
UNet linear LoRA (MatMul/Gemm). Spatial convolution, TE, DoRA, and LyCORIS adapters
must be rejected by the runtime unless a later manifest explicitly supports them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import onnx
from onnx import TensorProto, helper

FORMAT = 'rin-wai-lora-bank-v1'

def module_aliases(module: str) -> list[str]:
    """Exact aliases derived from SDXL's 3-block, 2-resnet-per-downblock topology."""
    aliases = {module, 'unet.' + module, 'lora_unet_' + module.replace('.', '_')}
    sgm = None
    match = re.fullmatch(r'down_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.+)', module)
    if match:
        block, kind, index, tail = match.groups()
        block, index = int(block), int(index)
        if block in range(3) and index in range(2):
            sgm = f'input_blocks.{block * 3 + index + 1}.{0 if kind == "resnets" else 1}.' + tail
    match = re.fullmatch(r'up_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.+)', module)
    if match:
        block, kind, index, tail = match.groups()
        block, index = int(block), int(index)
        if block in range(3) and index in range(3):
            sgm = f'output_blocks.{block * 3 + index}.{0 if kind == "resnets" else 1}.' + tail
    match = re.fullmatch(r'mid_block\.(resnets|attentions)\.(\d+)\.(.+)', module)
    if match:
        kind, index, tail = match.groups()
        if kind == 'resnets' and int(index) in (0, 1):
            sgm = f'middle_block.{int(index) * 2}.' + tail
        elif kind == 'attentions' and index == '0':
            sgm = 'middle_block.1.' + tail
    if sgm:
        if '.resnets.' in module:
            for old, new in [('time_emb_proj', 'emb_layers.1'), ('conv_shortcut', 'skip_connection'), ('conv1', 'in_layers.2'), ('conv2', 'out_layers.3')]:
                sgm = re.sub(r'\.' + re.escape(old) + r'(?=\.|$)', '.' + new, sgm)
        aliases.add('lora_unet_' + sgm.replace('.', '_'))
    for prefix in ('', 'unet.'):
        aliases.add('base_model.model.' + prefix + module)
    return sorted(aliases)

def tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField('t'):
                yield attr.t
            yield from attr.tensors
            if attr.HasField('g'):
                yield from tensors(attr.g)
            for child in attr.graphs:
                yield from tensors(child)

def linear_targets(model) -> list[dict]:
    init = {t.name: t for t in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    def constant_shape(name, depth=0):
        if depth > 8: return None
        if name in init: return list(init[name].dims), init[name].data_type
        node = producers.get(name)
        if node is None: return None
        if node.op_type == 'Identity': return constant_shape(node.input[0], depth + 1)
        if node.op_type == 'Transpose':
            resolved = constant_shape(node.input[0], depth + 1)
            if resolved is None: return None
            dims, dtype = resolved
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            perm = attrs.get('perm', list(range(len(dims) - 1, -1, -1)))
            return [dims[i] for i in perm], dtype
        return None
    result, seen = [], set()
    for index, node in enumerate(model.graph.node):
        if node.op_type not in ('MatMul', 'Gemm') or len(node.input) < 2: continue
        shape = constant_shape(node.input[1])
        if shape is None: continue
        dims, dtype = shape
        if len(dims) != 2 or dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16): continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == 'Gemm' and attrs.get('transA', 0):
            raise ValueError('Unsupported transposed activation at ' + node.name)
        module = node.name.rsplit('/', 1)[0].strip('/').replace('/', '.')
        if module.startswith('unet.'): module = module[5:]
        if not module or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9.]*', module):
            raise ValueError('Cannot derive exact module name: ' + node.name)
        if module in seen: raise ValueError('Duplicate target module: ' + module)
        seen.add(module)
        if node.op_type == 'Gemm' and attrs.get('transB', 0): out_features, in_features = dims
        else: in_features, out_features = dims
        result.append(dict(module=module, aliases=module_aliases(module), node_index=index,
                           node_name=node.name, op=node.op_type, input=node.input[0],
                           output=node.output[0], in_features=in_features, out_features=out_features,
                           dtype=dtype, gemm_alpha=float(attrs.get('alpha', 1.0))))
    if not result: raise ValueError('No supported linear LoRA targets')
    aliases = {}
    for item in result:
        for alias in item['aliases']:
            if alias in aliases and aliases[alias] != item['module']: raise ValueError('Alias collision: ' + alias)
            aliases[alias] = item['module']
    return result

def inject(model, capacity: int = 64):
    if capacity not in (8, 16, 32, 64, 128): raise ValueError('Unsupported rank capacity')
    targets = linear_targets(model)
    dtypes = {t['dtype'] for t in targets}
    if len(dtypes) != 1: raise ValueError('Mixed weight precision is not supported by one bank')
    dtype = dtypes.pop()
    original_inputs = [v.name for v in model.graph.input]
    original_outputs = [v.name for v in model.graph.output]
    names = {s for n in model.graph.node for s in list(n.input) + list(n.output)}
    if 'rin_lora_bank' in names or any(x.startswith('rin_lora__') for x in names):
        raise ValueError('Graph already has LoRA inputs')
    offset = 0
    for t in targets:
        t['rank_capacity'] = capacity
        t['a_offset'] = offset; offset += t['in_features'] * capacity
        t['b_offset'] = offset; offset += capacity * t['out_features']
    model.graph.input.append(helper.make_tensor_value_info('rin_lora_bank', dtype, [1, offset]))
    indexed = {t['node_index']: t for t in targets}
    nodes = []
    for index, node in enumerate(model.graph.node):
        t = indexed.get(index)
        if t is None:
            nodes.append(node); continue
        prefix = f'rin_lora__{len([x for x in nodes if x.name.startswith("rin_lora__")])}_{index}'
        original_output = node.output[0]
        node.output[0] = prefix + '_base'
        nodes.append(node)
        for label, start, shape in [('A', t['a_offset'], [t['in_features'], capacity]), ('B', t['b_offset'], [capacity, t['out_features']])]:
            end = start + shape[0] * shape[1]
            params = {'start': [start], 'end': [end], 'axes': [1], 'shape': shape}
            for key, vals in params.items():
                model.graph.initializer.append(helper.make_tensor(prefix + '_' + label + '_' + key, TensorProto.INT64, [len(vals)], vals))
            sliced = prefix + '_' + label + '_slice'; reshaped = prefix + '_' + label
            nodes.append(helper.make_node('Slice', ['rin_lora_bank', prefix + '_' + label + '_start', prefix + '_' + label + '_end', prefix + '_' + label + '_axes'], [sliced], name=sliced))
            nodes.append(helper.make_node('Reshape', [sliced, prefix + '_' + label + '_shape'], [reshaped], name=reshaped))
        nodes.append(helper.make_node('MatMul', [t['input'], prefix + '_A'], [prefix + '_rank'], name=prefix + '_down'))
        nodes.append(helper.make_node('MatMul', [prefix + '_rank', prefix + '_B'], [prefix + '_delta'], name=prefix + '_up'))
        nodes.append(helper.make_node('Add', [prefix + '_base', prefix + '_delta'], [original_output], name=prefix + '_sum'))
    del model.graph.node[:]; model.graph.node.extend(nodes)
    manifest = dict(format=FORMAT, bank_name='rin_lora_bank', bank_shape=[1, offset],
                    io_dtype='float32', graph_dtype=int(dtype), rank_capacity=capacity,
                    original_inputs=original_inputs, original_outputs=original_outputs,
                    layers=[{k: v for k, v in t.items() if k not in ('node_index', 'dtype', 'input', 'output')} for t in targets])
    return model, manifest

def prepare(source: Path, output: Path, capacity: int):
    source, output = source.resolve(), output.resolve()
    if output.exists(): raise FileExistsError('Refusing to overwrite: ' + str(output))
    model = onnx.load(str(source), load_external_data=False)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    refs = sorted({e.value for t in tensors(model.graph) for e in t.external_data if e.key == 'location'})
    for ref in refs:
        relative = Path(ref)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe external reference: ' + ref)
        p = source.parent / relative
        if not p.is_file() or p.stat().st_size <= 0: raise ValueError('Missing external tensor: ' + str(p))
    model, manifest = inject(model, capacity)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ref in refs:
        dest = output.parent / ref
        if dest.exists():
            if not dest.samefile(source.parent / ref): raise FileExistsError(str(dest))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(source.parent / ref, dest)
    output.write_bytes(model.SerializeToString())
    onnx.checker.check_model(str(output))
    manifest.update(source_onnx_sha256=source_sha, onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    source_external_files=len(refs), source_external_bytes=sum((source.parent / ref).stat().st_size for ref in refs))
    path = output.with_suffix('.lora.json')
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'onnx':str(output), 'manifest':str(path), 'layers':len(manifest['layers']),
                      'bank_float32_bytes':manifest['bank_shape'][1] * 4, 'rank_capacity':capacity}), flush=True)
    return manifest

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--out', required=True); ap.add_argument('--rank-capacity', type=int, default=64)
    a = ap.parse_args(); prepare(Path(a.source), Path(a.out), a.rank_capacity)
