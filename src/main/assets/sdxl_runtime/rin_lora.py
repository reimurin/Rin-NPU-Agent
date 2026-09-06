"""Safe, NumPy-only real LoRA loading for packed WAI QNN templates.
No pickle/torch dependency; no ignored tensors or lossy rank truncation.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json,math,os,re,struct,unicodedata,uuid
import numpy as np

class LoraError(ValueError): pass

@dataclass(frozen=True)
class Selection:
    name:str
    weight:float

@dataclass
class Plan:
    prompt:str
    negative:str
    contexts:dict
    inputs:dict
    metadata:dict

def parse_tags(text):
    if not isinstance(text,str) or len(text)>100000:raise LoraError('Prompt must be text of at most 100000 characters')
    items={};pattern=re.compile(r'<lora:([^<>]+)>',re.I)
    def replace(match):
        parts=match.group(1).rsplit(':',1);name=unicodedata.normalize('NFC',parts[0].strip());raw=parts[1].strip() if len(parts)==2 else '1'
        if name.lower().endswith('.safetensors'):name=name[:-12]
        if not name or name in ('.','..') or any(c in name for c in '/\\<>:') or any(ord(c)<32 for c in name):raise LoraError('Invalid LoRA name')
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',raw):raise LoraError('Invalid LoRA strength: '+raw)
        weight=float(raw)
        if not math.isfinite(weight) or not -2<=weight<=2:raise LoraError('LoRA strength must be finite and between -2 and 2')
        if name in items and items[name].weight!=weight:raise LoraError('Conflicting strengths for '+name)
        items[name]=Selection(name,weight);return ''
    clean=pattern.sub(replace,text)
    if re.search(r'<\s*lora\b',clean,re.I):raise LoraError('Malformed LoRA directive')
    return clean,list(items.values())

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def _unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise LoraError('Duplicate JSON key: '+k)
        result[k]=v
    return result

def _json(data):return json.loads(data,object_pairs_hook=_unique)
def _stamp(p):s=p.stat();return [s.st_size,s.st_mtime_ns]
def _inside(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):raise LoraError('Path escapes LoRA root')
    return p

class SafeWeights:
    def __init__(self,path):
        self.path=Path(path);self.stamp=_stamp(self.path)
        if self.path.is_symlink() or self.path.suffix.lower()!='.safetensors':raise LoraError('Expected regular .safetensors file')
        with self.path.open('rb') as f:
            b=f.read(8)
            if len(b)!=8:raise LoraError('Incomplete LoRA file')
            n=struct.unpack('<Q',b)[0]
            if not 2<=n<=8*1024*1024 or n+8>self.stamp[0]:raise LoraError('Invalid safetensors header size')
            self.header=_json(f.read(n).decode('utf-8'));self.start=8+n
        if not isinstance(self.header,dict) or len(self.header)>60000:raise LoraError('Invalid safetensors map')
        self.metadata=self.header.get('__metadata__',{})
        if not isinstance(self.metadata,dict) or any(not isinstance(v,str) for v in self.metadata.values()):raise LoraError('Invalid safetensors metadata')
        widths={'F16':2,'BF16':2,'F32':4};ranges=[];self.tensors={}
        for name,t in self.header.items():
            if name=='__metadata__':continue
            if not isinstance(t,dict):raise LoraError('Invalid tensor record')
            dims=t.get('shape');offsets=t.get('data_offsets');dtype=t.get('dtype')
            if dtype not in widths or not isinstance(dims,list) or len(dims)>8 or any(type(d)!=int or d<=0 for d in dims):raise LoraError('Unsupported dtype or tensor shape: '+name)
            if not isinstance(offsets,list) or len(offsets)!=2 or any(type(x)!=int for x in offsets):raise LoraError('Invalid tensor offsets')
            a,b=offsets
            if not 0<=a<b<=self.stamp[0]-self.start or b-a!=math.prod(dims)*widths[dtype]:raise LoraError('Incomplete/invalid tensor payload: '+name)
            ranges.append((a,b));self.tensors[name]=t
        end=0
        for a,b in sorted(ranges):
            if a!=end:raise LoraError('Tensor payload overlaps or has holes')
            end=b
        if not ranges or end!=self.stamp[0]-self.start:raise LoraError('Unexpected trailing or missing tensor payload')
        self.check_unchanged()
    def check_unchanged(self):
        if _stamp(self.path)!=self.stamp:raise LoraError('LoRA file changed during loading; wait for download to finish')
    def read(self,name):
        t=self.tensors[name];count=math.prod(t['shape']);dtype={'F16':'<f2','F32':'<f4','BF16':'<u2'}[t['dtype']]
        with self.path.open('rb') as f:
            f.seek(self.start+t['data_offsets'][0]);value=np.fromfile(f,dtype=dtype,count=count)
        if value.size!=count:raise LoraError('Short tensor read: '+name)
        if t['dtype']=='BF16':value=(value.astype(np.uint32)<<16).view(np.float32)
        else:value=value.astype(np.float32)
        if not np.isfinite(value).all():raise LoraError('Nonfinite LoRA weights: '+name)
        self.check_unchanged();return value.reshape(t['shape'])
    def pairs(self):
        used=set();result=[]
        suffixes=[('.lora_down.weight','.lora_up.weight'),('.lora_A.weight','.lora_B.weight'),('.lora_A.default.weight','.lora_B.default.weight'),('.lora.down.weight','.lora.up.weight')]
        for key,t in self.tensors.items():
            match=next(((a,b) for a,b in suffixes if key.endswith(a)),None)
            if match is None:continue
            a,b=match;module=key[:-len(a)];up=module+b;alpha=module+'.alpha'
            if up not in self.tensors:raise LoraError('Missing paired LoRA tensor: '+up)
            ds=t['shape'];us=self.tensors[up]['shape']
            if len(ds)==len(us)==4 and ds[2:]==us[2:]==[1,1]:ds=ds[:2];us=us[:2]
            if len(ds)!=2 or len(us)!=2 or ds[0]!=us[1]:raise LoraError('Only compatible linear/1x1 LoRA pairs are supported: '+module)
            if alpha in self.tensors and math.prod(self.tensors[alpha]['shape'])!=1:raise LoraError('Training alpha must be scalar')
            used.update([key,up]);used.update([alpha] if alpha in self.tensors else [])
            result.append({'alias':module,'down':key,'up':up,'alpha':alpha if alpha in self.tensors else None,'rank':ds[0],'in':ds[1],'out':us[0]})
        unknown=set(self.tensors)-used
        if not result or unknown:raise LoraError('Unsupported LoRA tensors (none ignored): '+', '.join(sorted(unknown)[:5]))
        return result

def validate_layout(manifest):
    if manifest.get('schema')!=1 or manifest.get('format')!='rin-lora-packed-linear-v1':raise LoraError('Unsupported LoRA template manifest')
    aliases={};modules={}
    for stage,spec in manifest['stages'].items():
        if stage not in ('encoder','decoder') or spec.get('input_name')!='rin_lora' or spec.get('dtype')!='float32':raise LoraError('Invalid LoRA stage')
        offset=0
        for m in spec['modules']:
            name=m['module'];i,o,rank=m['in_features'],m['out_features'],m['rank_capacity']
            if name in modules or any(type(v)!=int or v<=0 for v in [i,o,rank]) or rank>64:raise LoraError('Invalid/duplicate module')
            if m['a_offset']!=offset or m['a_elements']!=i*rank:raise LoraError('Invalid A layout')
            offset+=i*rank
            if m['b_offset']!=offset or m['b_elements']!=rank*o:raise LoraError('Invalid B layout')
            offset+=rank*o;modules[name]=(stage,m)
            for alias in [name,'lora_unet_'+name.removeprefix('unet.').replace('.','_'),'base_model.model.'+name]:
                if alias in aliases and aliases[alias]!=name:raise LoraError('Ambiguous module alias')
                aliases[alias]=name
        if not 0<offset==spec['elements']<=160_000_000:raise LoraError('Invalid stage buffer size')
    if set(manifest['stages'])!={'encoder','decoder'}:raise LoraError('Both WAI stages are required')
    return aliases,modules

def map_weights(manifest,selections,folder):
    aliases,modules=validate_layout(manifest);mapped={};evidence=[]
    files={}
    for p in folder.iterdir():
        if p.suffix.lower()=='.safetensors':files.setdefault(unicodedata.normalize('NFC',p.stem),[]).append(p)
    for selected in selections:
        if selected.weight==0:continue
        candidates=files.get(selected.name,[])
        if len(candidates)!=1:raise LoraError('Missing or ambiguous LoRA file: '+selected.name)
        path=_inside(folder,candidates[0].name);weights=SafeWeights(path);pairs=weights.pairs();sha=digest(path);weights.check_unchanged()
        seen_targets=set()
        for pair in pairs:
            target=aliases.get(pair['alias'])
            if target is None:raise LoraError('Template does not cover '+pair['alias']+'; text-encoder/conv/other architecture adapters are not silently ignored')
            if target in seen_targets:raise LoraError('Duplicate aliases map to one module in the same adapter: '+target)
            seen_targets.add(target)
            stage,module=modules[target]
            if (pair['in'],pair['out'])!=(module['in_features'],module['out_features']):raise LoraError('LoRA/base shape mismatch: '+target)
            entries=mapped.setdefault(target,[]);total=sum(x[1]['rank'] for x in entries)+pair['rank']
            if total>module['rank_capacity']:raise LoraError(f'{target}: combined rank {total} exceeds template capacity {module["rank_capacity"]}')
            entries.append((weights,pair,selected.weight))
        evidence.append({'name':selected.name,'sha256':sha,'weight':selected.weight,'modules':len(pairs),'file_bytes':path.stat().st_size})
    return mapped,evidence

def pack_weights(manifest,mapped,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);records={}
    for stage,spec in manifest['stages'].items():
        path=out/(stage+'.raw');buffer=np.memmap(path,mode='w+',dtype='<f4',shape=(spec['elements'],));buffer[:]=0
        try:
            for module in spec['modules']:
                i,o,capacity=module['in_features'],module['out_features'],module['rank_capacity'];offset=0
                av=buffer[module['a_offset']:module['a_offset']+module['a_elements']].reshape(i,capacity)
                bv=buffer[module['b_offset']:module['b_offset']+module['b_elements']].reshape(capacity,o)
                for source,pair,strength in mapped.get(module['module'],[]):
                    rank=pair['rank'];down=source.read(pair['down']).reshape(rank,i);up=source.read(pair['up']).reshape(o,rank)
                    alpha=float(source.read(pair['alpha']).reshape(-1)[0]) if pair['alpha'] else float(rank)
                    av[:,offset:offset+rank]=down.T;bv[offset:offset+rank,:]=up.T*(strength*alpha/rank);offset+=rank
                if not np.isfinite(av).all() or not np.isfinite(bv).all():raise LoraError('Weighted LoRA overflow')
            buffer.flush()
        finally:buffer._mmap.close()
        records[stage]={'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)}
    return records

def prepare(prompt,negative,base,width,height,log=print):
    clean,selections=parse_tags(prompt);neg,negative_tags=parse_tags(negative)
    if negative_tags:raise LoraError('Place LoRA controls in the positive prompt, not the negative prompt')
    active=[s for s in selections if s.weight!=0]
    if not active:return Plan(clean,neg,{}, {},{'active':False})
    base=Path(base).resolve()
    from rin_lora_partitioned import RELATIVE,prepare as prepare_partitioned
    new_manifest=base/RELATIVE/'lora_template.json'
    legacy_manifest=base/'lora_runtime/wai-linear-r32-1024/manifest.json'
    if new_manifest.is_file() or not legacy_manifest.is_file():
        return prepare_partitioned(clean,neg,base,width,height,active,log)
    folder=_inside(base,'Lora');template=_inside(base,'lora_runtime/wai-linear-r32-1024');mp=template/'manifest.json'
    if not mp.is_file():raise LoraError('LoRA-capable WAI model component is not installed. Prepare the LoRA model component first.')
    raw=mp.read_bytes()
    if len(raw)>8*1024*1024:raise LoraError('Oversized LoRA manifest')
    manifest=_json(raw.decode('utf-8'))
    if not manifest.get('compiled') or manifest.get('base_model')!='WAI-illustrious-SDXL-v170' or manifest.get('resolution')!=[width,height]:raise LoraError('LoRA template base/resolution mismatch')
    if manifest.get('format')=='rin-lora-module-banks-v1':
        from rin_lora_module_banks import prepare_modules
        return prepare_modules(clean,neg,base,manifest,raw,active,folder,template,log)
    mapped,evidence=map_weights(manifest,active,folder);contexts={};cache=_inside(base,'.rin_lora/cache');cache.mkdir(parents=True,exist_ok=True)
    for stage,spec in manifest['stages'].items():
        path=_inside(template,spec['context'])
        if not path.is_file() or path.stat().st_size!=spec['context_bytes']:raise LoraError('Incomplete LoRA context: '+stage)
        stamp={'path':str(path),'stamp':_stamp(path),'sha256':spec['context_sha256']};marker=cache/('verified_'+stage+'.json')
        valid=False
        try:valid=_json(marker.read_text())==stamp
        except (OSError,ValueError):pass
        if not valid:
            log('[LoRA] verifying '+stage+' model')
            if digest(path)!=spec['context_sha256']:raise LoraError('LoRA context integrity mismatch: '+stage)
            marker.write_text(json.dumps(stamp))
        contexts[stage]=str(path)
    key=hashlib.sha256(json.dumps({'manifest':hashlib.sha256(raw).hexdigest(),'files':evidence},sort_keys=True).encode()).hexdigest()
    dest=cache/key;pointer=cache/(key+'.pointer.json');records=None
    try:
        directory=_json(pointer.read_text())['directory']
        if not isinstance(directory,str) or not re.fullmatch(re.escape(key)+r'_[0-9a-f]{32}',directory):raise LoraError('Invalid cache pointer')
        candidate_dir=_inside(cache,directory)
        if candidate_dir.parent!=cache.resolve() or candidate_dir.is_symlink():raise LoraError('Invalid cache directory')
        dest=candidate_dir
    except (OSError,ValueError,KeyError,TypeError):pass
    try:
        saved=_json((dest/'ready.json').read_text());candidate=saved['packs']
        valid=isinstance(candidate,dict) and set(candidate)==set(manifest['stages']) and saved.get('files')==evidence
        if valid:
            for stage,item in candidate.items():
                if item.get('file')!=stage+'.raw' or item.get('bytes')!=manifest['stages'][stage]['elements']*4:valid=False;break
                file=_inside(dest,item['file'])
                if not file.is_file() or file.stat().st_size!=item['bytes'] or digest(file)!=item.get('sha256'):valid=False;break
        if valid:records=candidate
    except (OSError,ValueError,KeyError,TypeError,AttributeError):pass
    if records is None:
        log('[LoRA] mapping '+str(len(mapped))+' real WAI modules; preparing cache')
        dest=cache/(key+'_'+uuid.uuid4().hex);records=pack_weights(manifest,mapped,dest)
        (dest/'ready.json').write_text(json.dumps({'files':evidence,'packs':records},indent=2),encoding='utf-8')
        # Stable pointer names are written only after all raw files validate.
        pointer=cache/(key+'.pointer.json');pending_pointer=pointer.with_name(pointer.name+'.'+uuid.uuid4().hex+'.pending')
        pending_pointer.write_text(json.dumps({'directory':dest.name}));os.replace(pending_pointer,pointer)
    inputs={s:f'{manifest["stages"][s]["input_name"]}:={dest/r["file"]}' for s,r in records.items()}
    return Plan(clean,neg,contexts,inputs,{'active':True,'template_id':manifest['template_id'],'files':evidence,'mapped_modules':len(mapped),'cache_dir':str(dest)})
