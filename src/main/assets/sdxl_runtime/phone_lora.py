"""Phone-side dynamic LoRA adapter preparation; numpy/stdlib only.
No pickle/torch loading. Unmapped tensors cause a hard compatibility error.
"""
from __future__ import annotations
import hashlib,json,math,os,re,struct,unicodedata,uuid
from pathlib import Path
import numpy as np

TEMPLATE_REL='context/lora/wai-v170-sm8750-lora-r64-1024-v1/lora_template.json'
class LoraError(ValueError):pass

def _unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise LoraError('Duplicate JSON field: '+k)
        d[k]=v
    return d

def _json(path,limit=16*1024*1024):
    p=Path(path)
    if p.stat().st_size>limit:raise LoraError('Metadata exceeds size limit')
    return json.loads(p.read_text(encoding='utf-8-sig'),object_pairs_hook=_unique)

def parse_tags(prompt):
    controls={};number=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)')
    def replace(m):
        fields=m.group(1).rsplit(':',1);name=unicodedata.normalize('NFC',fields[0].strip())
        if name.lower().endswith('.safetensors'):name=name[:-12]
        raw=fields[1].strip() if len(fields)==2 else '1'
        if not number.fullmatch(raw):raise LoraError('Invalid LoRA strength: '+raw)
        strength=float(raw)
        if not math.isfinite(strength) or not -2<=strength<=2:raise LoraError('LoRA strength must be between -2 and 2')
        if not name or name in ('.','..') or any(c in name for c in '/\\<>:') or any(ord(c)<32 for c in name):raise LoraError('Invalid LoRA name')
        if name in controls and controls[name]!=strength:raise LoraError('Conflicting duplicate LoRA: '+name)
        controls[name]=strength;return ''
    clean=re.sub(r'<lora:([^<>]+)>',replace,prompt,flags=re.I)
    if re.search(r'<\s*lora\b',clean,re.I):raise LoraError('Malformed LoRA tag')
    return clean,controls

class SafeWeights:
    widths={'F16':2,'BF16':2,'F32':4}
    def __init__(self,path):
        self.path=Path(path)
        if self.path.is_symlink() or not self.path.is_file():raise LoraError('Not a regular LoRA file')
        self.stat=self.path.stat()
        if not 10<=self.stat.st_size<=4*2**30:raise LoraError('LoRA file size outside supported range')
        with self.path.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0]
            if not 2<=size<=8*2**20 or size+8>self.stat.st_size:raise LoraError('Incomplete Safetensors header')
            self.tensors=json.loads(f.read(size).decode('utf-8'),object_pairs_hook=_unique)
        if not isinstance(self.tensors,dict) or len(self.tensors)>60000:raise LoraError('Invalid or excessive tensor map')
        self.offset=size+8;self.meta=self.tensors.pop('__metadata__',{})
        if not isinstance(self.meta,dict) or any(not isinstance(v,str) for v in self.meta.values()):raise LoraError('Invalid metadata')
        family=self.meta.get('ss_base_model_version','').lower()
        if family and not any(x in family for x in ['sdxl','illustrious','pony']):raise LoraError('This adapter is not declared as SDXL')
        ranges=[]
        for name,t in self.tensors.items():
            if not isinstance(t,dict):raise LoraError('Invalid tensor record')
            shape=t.get('shape');off=t.get('data_offsets');dtype=t.get('dtype')
            if dtype not in self.widths or not isinstance(shape,list) or len(shape)>4 or any(type(x)!=int or x<=0 or x>65536 for x in shape):raise LoraError('Unsupported tensor: '+name)
            if not isinstance(off,list) or len(off)!=2 or any(type(x)!=int for x in off):raise LoraError('Invalid tensor offsets')
            a,b=off
            if a<0 or b<=a or b+self.offset>self.stat.st_size or b-a!=math.prod(shape)*self.widths[dtype]:raise LoraError('Incomplete or invalid tensor payload: '+name)
            ranges.append((a,b))
        end=0
        for a,b in sorted(ranges):
            if a!=end:raise LoraError('Overlapping or missing tensor data')
            end=b
        if not ranges or end+self.offset!=self.stat.st_size:raise LoraError('Unexpected Safetensors payload')
    def read(self,name):
        t=self.tensors[name];a,b=t['data_offsets'];dtype=t['dtype']
        with self.path.open('rb') as f:f.seek(self.offset+a);raw=f.read(b-a)
        if len(raw)!=b-a:raise LoraError('LoRA changed while reading')
        if dtype=='BF16':out=(np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32)
        else:out=np.frombuffer(raw,'<f2' if dtype=='F16' else '<f4').astype(np.float32)
        if not np.isfinite(out).all():raise LoraError('Non-finite LoRA weights: '+name)
        return out.reshape(t['shape'])
    def stable(self):
        s=self.path.stat()
        if (s.st_size,s.st_mtime_ns)!=(self.stat.st_size,self.stat.st_mtime_ns):raise LoraError('LoRA file is still changing')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*2**20),b''):h.update(data)
    return h.hexdigest()

def _atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def _safe_child(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or path==root:raise LoraError('Path escapes LoRA component directory')
    return path

def load_template(base,resolution):
    file=Path(base)/TEMPLATE_REL
    if not file.is_file():raise LoraError('LoRA runtime model component is not installed. Install it from LoRA management first.')
    t=_json(file)
    if t.get('schema')!=1 or not t.get('complete') or t.get('resolution')!=list(resolution):raise LoraError('LoRA component version or resolution mismatch')
    if set(t.get('graphs',{}))!={'encoder','decoder'}:raise LoraError('Incomplete LoRA component')
    modules={};aliases={};contexts={};fingerprints={};capacity=t.get('rank_capacity')
    if capacity not in (16,32,64,128):raise LoraError('Unsupported compiled rank capacity')
    for graph,spec in t['graphs'].items():
        ctx=_safe_child(file.parent,spec['context_file'])
        if not ctx.is_file() or ctx.stat().st_size!=spec['context_bytes']:raise LoraError('LoRA context missing or incomplete: '+graph)
        if not re.fullmatch('[0-9a-f]{64}',spec['context_sha256']):raise LoraError('Invalid model checksum')
        contexts[graph]=str(ctx);fingerprints[graph]=[ctx.stat().st_size,ctx.stat().st_mtime_ns,spec['context_sha256']]
        seen=set()
        for layer in spec['layers']:
            module=layer['module']
            if module in modules or len(modules)>2048:raise LoraError('Duplicate or excessive modules')
            inf,outf=layer['in_features'],layer['out_features']
            if not 1<=inf<=16384 or not 1<=outf<=16384 or layer['capacity']!=capacity or layer['a_shape']!=[inf,capacity] or layer['b_shape']!=[capacity,outf]:raise LoraError('Invalid model matrix shape')
            for key in ('a','b'):
                name=layer[key]
                if not re.fullmatch(r'rin_lora_\d{4}_[ab]',name) or name in seen:raise LoraError('Invalid or duplicate input name')
                seen.add(name)
            modules[module]=(graph,layer)
            for alias in layer['aliases']+[module]:
                if alias in aliases and aliases[alias]!=module:raise LoraError('Ambiguous module alias')
                aliases[alias]=module
    integrity=Path(base)/'.rin_lora/integrity.json';tag=sha(file)
    try:prior=_json(integrity)
    except (OSError,ValueError):prior={}
    if prior!={'template':tag,'files':fingerprints}:
        for graph,path in contexts.items():
            if sha(path)!=fingerprints[graph][2]:raise LoraError('LoRA model checksum mismatch: '+graph)
        _atomic_json(integrity,{'template':tag,'files':fingerprints})
    return t,modules,aliases,contexts,tag

def resolve_pairs(weights,aliases):
    suffixes=[('.lora_down.weight','.lora_up.weight'),('.lora_A.weight','.lora_B.weight'),('.lora_A.default.weight','.lora_B.default.weight'),('.lora.down.weight','.lora.up.weight'),('.lora_linear_layer.down.weight','.lora_linear_layer.up.weight')]
    result={};used=set()
    for key in weights.tensors:
        match=next(((a,b) for a,b in suffixes if key.endswith(a)),None)
        if not match:continue
        a,b=match;prefix=key[:-len(a)];other=prefix+b;lookup=prefix
        if lookup.startswith('base_model.model.'):lookup=lookup[len('base_model.model.'):]
        if lookup not in aliases:
            if lookup.startswith(('text_encoder','lora_te','te1','te2')):raise LoraError('This component currently supports UNet LoRA only; the file also modifies a text encoder: '+lookup)
            raise LoraError('Unsupported attachment point (not ignored): '+lookup)
        module=aliases[lookup]
        if module in result or other not in weights.tensors:raise LoraError('Duplicate module or missing LoRA pair: '+module)
        alpha=prefix+'.alpha';used.update((key,other))
        if alpha in weights.tensors:used.add(alpha)
        result[module]=(key,other,alpha if alpha in weights.tensors else None)
    extra=set(weights.tensors)-used
    if not result or extra:raise LoraError('Unsupported LoRA keys: '+str(sorted(extra)[:5]))
    return result

class Session:
    def __init__(self,contexts,entries,report):self.contexts=contexts;self.entries=entries;self.report=report
    def extend_input_list(self,context,path):
        graph=next((g for g,p in self.contexts.items() if Path(p).resolve()==Path(context).resolve()),None)
        if graph is None:return
        p=Path(path);lines=p.read_text().splitlines();extra=self.entries[graph];rows=[]
        for line in lines:
            if not line.strip() or line.lstrip().startswith('#'):continue
            tokens=line.split();tokens=[x for x in tokens if x.split(':=',1)[0] not in extra]
            if any(':=' not in x for x in tokens):raise LoraError('LoRA graph requires named original inputs')
            rows.append(' '.join(tokens+[name+':='+value for name,value in extra.items()]))
        if not rows:raise LoraError('Empty LoRA input list')
        body='\n'.join(rows)+'\n'
        if p.read_text()!=body:
            tmp=p.with_name(p.name+'.lora.tmp');tmp.write_text(body);os.replace(tmp,p)

def prepare_request(prompt,negative,base,resolution=(1024,1024),log=print):
    prompt,selections=parse_tags(prompt);negative,neg=parse_tags(negative)
    for name,value in neg.items():
        if name in selections and selections[name]!=value:raise LoraError('Conflicting positive/negative LoRA strengths: '+name)
        selections[name]=value
    active={name:value for name,value in selections.items() if value!=0}
    if not active:return prompt,negative,None
    base=Path(base).resolve();folder=(base/'Lora').resolve()
    if not folder.is_relative_to(base):raise LoraError('LoRA folder escapes runtime directory')
    template,modules,aliases,contexts,template_sha=load_template(base,resolution)
    capacity=template['rank_capacity'];readers=[];pairs={};identity=[]
    for name,strength in active.items():
        candidates=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower()=='.safetensors' and unicodedata.normalize('NFC',p.stem)==name] if folder.is_dir() else []
        if len(candidates)!=1:raise LoraError('LoRA file missing or ambiguous: '+name)
        path=candidates[0]
        if path.is_symlink() or path.resolve().parent!=folder:raise LoraError('LoRA file outside selected directory')
        weights=SafeWeights(path);mapped=resolve_pairs(weights,aliases);reader=len(readers);readers.append(weights)
        for module,keys in mapped.items():
            _,spec=modules[module];down=weights.tensors[keys[0]]['shape'];up=weights.tensors[keys[1]]['shape']
            if len(down)==4 and down[2:]==[1,1]:down=down[:2]
            if len(up)==4 and up[2:]==[1,1]:up=up[:2]
            if len(down)!=2 or len(up)!=2 or down[0]!=up[1] or down[1]!=spec['in_features'] or up[0]!=spec['out_features']:raise LoraError('LoRA matrix shape mismatch: '+module)
            alpha=float(weights.read(keys[2]).reshape(-1)[0]) if keys[2] else down[0]
            if keys[2] and math.prod(weights.tensors[keys[2]]['shape'])!=1:raise LoraError('Alpha must be scalar')
            pairs.setdefault(module,[]).append((reader,keys,down[0],strength*alpha/down[0]))
        identity.append({'name':name,'strength':strength,'sha256':sha(path),'modules':len(mapped)})
        weights.stable()
    for module,parts in pairs.items():
        rank=sum(p[2] for p in parts)
        if rank>capacity:raise LoraError(f'Combined rank {rank} exceeds compiled capacity {capacity}: {module}')
    key=hashlib.sha256(json.dumps({'schema':1,'template':template_sha,'adapters':identity},sort_keys=True).encode()).hexdigest()
    cache=base/'.rin_lora/cache'/key;metadata=cache/'prepared.json'
    if metadata.is_file():
        saved=_json(metadata)
        entries={};valid=saved.get('key')==key
        for graph,items in saved.get('entries',{}).items():
            entries[graph]={}
            for name,item in items.items():
                file=_safe_child(cache,item['file']);valid=valid and file.is_file() and file.stat().st_size==item['bytes']
                if valid and sha(file)!=item['sha256']:valid=False
                entries[graph][name]=str(file)
        expected_inputs={g:{layer[k] for layer in spec['layers'] for k in ['a','b']} for g,spec in template['graphs'].items()}
        valid=valid and set(entries)==set(contexts) and all(set(entries.get(g,{}))==names for g,names in expected_inputs.items())
        if valid:return prompt,negative,Session(contexts,entries,saved['report'])
        raise LoraError('LoRA cache is incomplete or corrupt; clear LoRA cache and retry')
    temp=cache.with_name(key+'.pending-'+uuid.uuid4().hex);temp.mkdir(parents=True)
    entries={};stored={};zero_files={}
    for graph,model in template['graphs'].items():
        stored[graph]={}
        for spec in model['layers']:
            inf,outf=spec['in_features'],spec['out_features'];parts=pairs.get(spec['module'],[])
            aa=np.zeros((inf,capacity),np.float32);bb=np.zeros((capacity,outf),np.float32);offset=0
            for reader,keys,rank,scale in parts:
                aa[:,offset:offset+rank]=readers[reader].read(keys[0]).reshape(rank,inf).T
                bb[offset:offset+rank,:]=readers[reader].read(keys[1]).reshape(outf,rank).T*scale
                offset+=rank
            for letter,array in [('a',aa),('b',bb)]:
                if not np.isfinite(array).all():raise LoraError('Scaled LoRA overflow')
                name=spec[letter];filename=graph+'_'+name+'.raw'
                if not parts:
                    shape=tuple(array.shape)
                    filename=zero_files.setdefault(shape,'zero_'+'x'.join(map(str,shape))+'.raw')
                file=temp/filename
                if not file.exists():array.astype('<f4',copy=False).tofile(file)
                stored[graph][name]={'file':filename,'bytes':array.nbytes,'sha256':sha(file)}
    for weights in readers:weights.stable()
    report={'model_id':template['model_id'],'adapters':identity,'matched_modules':len(pairs),'rank_capacity':capacity,'route':'dynamic_weights'}
    _atomic_json(temp/'prepared.json',{'key':key,'entries':stored,'report':report})
    cache.parent.mkdir(parents=True,exist_ok=True);os.replace(temp,cache)
    for graph,items in stored.items():entries[graph]={name:str(cache/item['file']) for name,item in items.items()}
    log('[LoRA ready] '+json.dumps(report,ensure_ascii=False))
    return prompt,negative,Session(contexts,entries,report)
