"""Isolated 1.6 prototype: tag parsing and safe LoRA inspection, not APP integration."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, math, re, struct, unicodedata
import numpy as np

class LoraError(ValueError):
    pass

@dataclass(frozen=True)
class Span:
    text: str
    weight: float = 1.0

@dataclass(frozen=True)
class Lora:
    name: str
    weight: float = 1.0

@dataclass(frozen=True)
class Prompt:
    text: str
    spans: tuple[Span, ...]
    loras: tuple[Lora, ...]

_NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
_CONTROL = re.compile(r'<lora:([^<>]+)>', re.IGNORECASE)


def number(text: str, *, lora: bool = False) -> float:
    text = text.strip()
    if not re.fullmatch(_NUM, text):
        raise LoraError(f'Invalid weight: {text!r}')
    value = float(text)
    low = -2.0 if lora else 0.0
    if not math.isfinite(value) or not low <= value <= 2.0:
        raise LoraError(f'Weight outside [{low}, 2]: {text}')
    return value


def _unescape(text: str) -> str:
    return re.sub(r'\\([\\():,<>])', r'\1', text)


def _colon(text: str) -> int:
    depth, escaped, last = 0, False, -1
    for i, c in enumerate(text):
        if escaped: escaped = False; continue
        if c == '\\': escaped = True; continue
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == ':' and depth == 0: last = i
    return last


def _parse_groups(text: str, scale: float = 1.0, depth: int = 0) -> list[Span]:
    if depth > 12: raise LoraError('Too many nested weights')
    parts, buf, i = [], [], 0
    def flush():
        if buf: parts.append(Span(_unescape(''.join(buf)), scale)); buf.clear()
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text): buf.extend(text[i:i+2]); i += 2; continue
        if c == ')': raise LoraError('Unmatched closing parenthesis')
        if c != '(': buf.append(c); i += 1; continue
        flush(); start = i + 1; level, escaped, j = 1, False, i + 1
        while j < len(text) and level:
            v = text[j]
            if escaped: escaped = False
            elif v == '\\': escaped = True
            elif v == '(': level += 1
            elif v == ')': level -= 1
            j += 1
        if level: raise LoraError('Unclosed weighted group')
        inner = text[start:j-1]; at = _colon(inner)
        if at >= 0:
            factor = number(inner[at+1:]); inner = inner[:at]
        else: factor = 1.1
        if not inner.strip(): raise LoraError('Empty weighted group')
        combined = scale * factor
        if not 0 <= combined <= 4: raise LoraError('Nested weight outside [0, 4]')
        parts.extend(_parse_groups(inner, combined, depth + 1)); i = j
    flush(); return parts


def parse_prompt(text: str) -> Prompt:
    if not isinstance(text, str) or len(text) > 100000: raise LoraError('Invalid prompt length/type')
    loras, seen = [], {}
    def control(match):
        fields = match.group(1).rsplit(':', 1)
        name = unicodedata.normalize('NFC', fields[0].strip())
        weight = number(fields[1], lora=True) if len(fields) == 2 else 1.0
        if not name or name in ('.', '..') or any(c in name for c in '/\\<>:') or any(ord(c) < 32 for c in name):
            raise LoraError('Invalid LoRA name')
        if name.lower().endswith('.safetensors'): name = name[:-12]
        if not name: raise LoraError('Empty LoRA alias')
        if name in seen:
            if seen[name] != weight: raise LoraError(f'Conflicting duplicate LoRA: {name}')
        else: seen[name] = weight; loras.append(Lora(name, weight))
        return ''
    # LoRA directives are global adapter controls, never multiplied by word emphasis.
    for match in _CONTROL.finditer(text):
        if match.start() and text[match.start()-1] == '\\': raise LoraError('Escaped LoRA directive unsupported')
    clean = _CONTROL.sub(control, text)
    if re.search(r'<\s*lora\b', clean, re.I): raise LoraError('Malformed LoRA directive')
    spans, start, depth, escaped = [], 0, 0, False
    segments = []
    for i, c in enumerate(clean):
        if escaped: escaped = False; continue
        if c == '\\': escaped = True; continue
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c in ',，' and depth == 0: segments.append((clean[start:i], c)); start = i + 1
        if depth < 0: raise LoraError('Unmatched closing parenthesis')
    if depth: raise LoraError('Unclosed weighted group')
    segments.append((clean[start:], ''))
    for segment, delimiter in segments:
        at = _colon(segment); scale = 1.0
        if at >= 0 and re.fullmatch(_NUM, segment[at+1:].strip()):
            phrase = segment[:at].strip()
            # Preserve URLs and ratios; bare shorthand is comma-delimited text only.
            if phrase and not re.fullmatch(r'[\d.]+', phrase) and ':' not in phrase and '/' not in phrase:
                scale = number(segment[at+1:]); segment = segment[:at]
        spans.extend(_parse_groups(segment, scale))
        if delimiter: spans.append(Span(delimiter))
    return Prompt(''.join(s.text for s in spans), tuple(spans), tuple(loras))


def runtime_paths(runtime_base: Path, create: bool = False) -> dict[str, Path]:
    base = Path(runtime_base).resolve()
    paths = {'originals': base/'Lora', 'cache': base/'.rin_lora'/'cache'}
    for p in paths.values():
        if not p.resolve().is_relative_to(base): raise LoraError('LoRA directory escapes runtime root')
    if create:
        for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    return paths


def _unique(pairs):
    out = {}
    for k, v in pairs:
        if k in out: raise LoraError(f'Duplicate JSON field: {k}')
        out[k] = v
    return out


def read_header(path: Path, max_header: int = 8*1024*1024) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise LoraError('Not a regular safetensors file')
    before = path.stat()
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8: raise LoraError('Incomplete safetensors header')
        length, = struct.unpack('<Q', raw)
        if not 2 <= length <= max_header or length + 8 > before.st_size: raise LoraError('Invalid header length')
        header = json.loads(f.read(length).decode('utf-8'), object_pairs_hook=_unique)
    if not isinstance(header, dict) or len(header) > 60000: raise LoraError('Invalid tensor map')
    meta = header.get('__metadata__', {})
    if not isinstance(meta, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in meta.items()):
        raise LoraError('Metadata must contain strings')
    widths = {'F16':2, 'BF16':2, 'F32':4, 'F64':8, 'I64':8, 'I32':4, 'I16':2, 'I8':1, 'U8':1, 'BOOL':1}
    ranges = []
    for name, tensor in header.items():
        if name == '__metadata__': continue
        if not isinstance(tensor, dict): raise LoraError('Invalid tensor record')
        shape, offsets, dtype = tensor.get('shape'), tensor.get('data_offsets'), tensor.get('dtype')
        if not isinstance(shape,list) or len(shape)>8 or any(type(d)!=int or d<=0 for d in shape): raise LoraError('Invalid shape')
        if dtype not in widths or not isinstance(offsets,list) or len(offsets)!=2 or any(type(x)!=int for x in offsets): raise LoraError('Invalid dtype/offsets')
        begin,end = offsets
        if not 0 <= begin < end <= before.st_size-length-8 or end-begin != math.prod(shape)*widths[dtype]: raise LoraError('Tensor payload size mismatch')
        ranges.append((begin,end))
    end = 0
    for begin, stop in sorted(ranges):
        if begin != end: raise LoraError('Overlapping tensor data or holes')
        end = stop
    if not ranges or end != before.st_size-length-8: raise LoraError('Incomplete or trailing payload')
    after = path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise LoraError('File is still changing')
    return header


def inspect_lora(path: Path) -> dict:
    header = read_header(path); tensors = {k:v for k,v in header.items() if k!='__metadata__'}
    meta = header.get('__metadata__', {}); base = meta.get('ss_base_model_version','').lower()
    if base and not any(k in base for k in ('sdxl','illustrious')): raise LoraError('Declared base model is not SDXL')
    pairs, used = [], set()
    for down, tensor in tensors.items():
        if down.endswith('.lora_down.weight'): prefix=down[:-17]; up=prefix+'.lora_up.weight'; fmt='kohya'
        elif down.endswith('.lora_A.weight'): prefix=down[:-14]; up=prefix+'.lora_B.weight'; fmt='peft'
        else: continue
        if up not in tensors: raise LoraError(f'Missing up matrix: {up}')
        a,b=tensor['shape'],tensors[up]['shape']
        if len(a) not in (2,4) or len(b)!=len(a) or a[0]!=b[1]: raise LoraError('Inconsistent LoRA rank')
        if len(a)==4 and (a[2:]!=[1,1] or b[2:]!=[1,1]): raise LoraError('Spatial convolution LoRA not yet supported')
        if any(tensors[k]['dtype'] not in ('F16','F32','BF16') for k in (down,up)): raise LoraError('Unsupported adapter dtype')
        if a[0]>256: raise LoraError('Rank exceeds prototype limit')
        alpha=prefix+'.alpha';used.update((down,up))
        if alpha in tensors:
            if math.prod(tensors[alpha]['shape'])!=1: raise LoraError('Alpha must be scalar')
            used.add(alpha)
        pairs.append({'module':prefix,'format':fmt,'rank':a[0],'down':down,'up':up,'alpha':alpha if alpha in tensors else None})
    extra=set(tensors)-used
    if not pairs or extra: raise LoraError('Unrecognized or unsupported adapter keys: '+str(sorted(extra)[:4]))
    return {'name':Path(path).stem, 'status':'needs_qnn_template', 'declared_base':base or 'unknown',
            'pairs':pairs, 'qnn_ready':False, 'metadata':meta}


def scan_loras(runtime_base: Path) -> list[dict]:
    root=runtime_paths(runtime_base)['originals'];out=[]
    if not root.is_dir(): return out
    for p in sorted(root.iterdir(),key=lambda p:p.name.casefold()):
        if p.suffix.lower()!='.safetensors': continue
        try:
            item=inspect_lora(p);item['path']=p.name
        except (LoraError, ValueError, OSError) as e:
            item={'name':p.stem,'path':p.name,'status':'invalid_or_incomplete','error':str(e),'qnn_ready':False}
        out.append(item)
    return out


def lora_delta(down, up, training_alpha: float | None, strength: float = 1.0):
    a,b=np.asarray(down,dtype=np.float32),np.asarray(up,dtype=np.float32)
    shape=None
    if a.ndim==b.ndim==4 and a.shape[2:]==b.shape[2:]==(1,1):
        shape=(b.shape[0],a.shape[1],1,1);a=a[:,:,0,0];b=b[:,:,0,0]
    if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1] or not a.shape[0]: raise LoraError('Invalid adapter matrices')
    alpha=float(training_alpha) if training_alpha is not None else float(a.shape[0])
    if not math.isfinite(alpha) or not math.isfinite(strength) or not np.isfinite(a).all() or not np.isfinite(b).all(): raise LoraError('Nonfinite adapter')
    out=(b@a)*(alpha/a.shape[0])*strength
    if not np.isfinite(out).all(): raise LoraError('Adapter overflow')
    return out.reshape(shape) if shape else out
