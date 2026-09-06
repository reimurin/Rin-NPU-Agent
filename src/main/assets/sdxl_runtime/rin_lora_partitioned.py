"""Real-LoRA preparation and sequential QNN execution for verified WAI partitions."""
from pathlib import Path
from dataclasses import dataclass
import hashlib,json,math,os,re,shutil,tempfile,uuid
import numpy as np
from rin_lora import LoraError,Plan,SafeWeights,_json,_stamp,digest
from rin_tensor_io import named_input,write_input_rows,resolve_output,MANIFEST_NAME,prepare_output_dir

FORMAT='rin-wai-lora-partitioned-inputs-v1'
MODEL_ID='wai-v170-sm8750-lora-r64-1024-partitioned-v1'
RELATIVE='context/lora/'+MODEL_ID

def inside(root,name):
    root=Path(root).resolve();resolved=(root/name).resolve()
    if resolved==root or not resolved.is_relative_to(root):raise LoraError('Path escapes component root')
    return resolved

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(obj,stream,ensure_ascii=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def read_json(path,limit=16*1024*1024):
    path=Path(path)
    if path.stat().st_size>limit:raise LoraError('Oversized component metadata')
    return _json(path.read_text(encoding='utf-8-sig'))

def shape_valid(shape):
    return isinstance(shape,list) and 1<=len(shape)<=4 and all(type(x)==int and 0<x<=65536 for x in shape) and math.prod(shape)*4<=512*1024*1024

def validate(manifest):
    if not isinstance(manifest,dict) or manifest.get('schema')!=1 or manifest.get('format')!=FORMAT or manifest.get('model_id')!=MODEL_ID:raise LoraError('Unsupported partitioned LoRA component')
    if manifest.get('complete') is not True or manifest.get('soc')!=69 or manifest.get('dsp')!=79 or manifest.get('base_model')!='WAI-illustrious-SDXL-v170':raise LoraError('Incomplete or wrong-target LoRA component')
    graphs=manifest.get('graphs');capacity=manifest.get('rank_capacity')
    if not isinstance(graphs,dict) or set(graphs)!={'encoder','decoder'} or capacity!=64 or manifest.get('resolution')!=[1024,1024]:raise LoraError('Wrong LoRA resolution/rank')
    modules={};aliases={};part_ids=set()
    for stage,spec in graphs.items():
        if not isinstance(spec.get('parts'),list) or not 1<=len(spec['parts'])<=8:raise LoraError('Invalid graph partition count')
        available=set(spec['original_inputs']);known_shapes={};flat=[]
        if len(available)!=len(spec['original_inputs']):raise LoraError('Repeated original graph input')
        for index,part in enumerate(spec['parts']):
            pid=part.get('id')
            if pid!=stage+'_p'+str(index) or pid in part_ids or part.get('context_file')!=pid+'.bin':raise LoraError('Invalid partition identity')
            part_ids.add(pid)
            if type(part.get('context_bytes'))!=int or not 1<=part['context_bytes']<=4*1024**3 or not re.fullmatch('[0-9a-f]{64}',str(part.get('context_sha256',''))):raise LoraError('Invalid partition file metadata')
            used_inputs=set();used_outputs=set()
            for label in ('input','output'):
                records=part.get(label+'s');bindings=part.get(label+'_bindings')
                if not isinstance(records,list) or not records or not isinstance(bindings,list) or len(records)!=len(bindings):raise LoraError('Incomplete partition interface')
                wanted={x['name']:x['shape'] for x in records};used=set()
                if len(wanted)!=len(records):raise LoraError('Duplicate logical interface')
                for binding in bindings:
                    logical=binding.get('logical');name=binding.get('name');shape=binding.get('shape');dims=binding.get('qnn_shape');axes=binding.get('to_qnn_axes')
                    if logical in used or logical not in wanted or name not in (logical,logical+'_permute') or not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',str(name)):raise LoraError('Ambiguous semantic tensor binding')
                    used.add(logical)
                    if shape!=wanted[logical] or not shape_valid(shape) or not shape_valid(dims) or not isinstance(axes,list) or sorted(axes)!=list(range(len(shape))) or [shape[a] for a in axes]!=dims:raise LoraError('Invalid partition tensor layout: '+str(name))
                    if axes!=list(range(len(shape))) and not (len(shape)==4 and axes==[0,2,3,1]):raise LoraError('Unvalidated layout transformation')
                    if binding.get('qnn_dtype') not in ('QNN_DATATYPE_FLOAT_16','QNN_DATATYPE_FLOAT_32'):raise LoraError('Unsupported partition dtype')
                    if label=='input':
                        if logical not in available:raise LoraError('Graph dependency unavailable: '+logical)
                        if logical in known_shapes and known_shapes[logical]!=shape:raise LoraError('Partition boundary shape disagrees')
                        used_inputs.add(name)
                    else:used_outputs.add(name);known_shapes[logical]=shape
                if used!=set(wanted):raise LoraError('Missing logical tensor binding')
            for layer in part.get('layers',[]):
                module=layer.get('module');inf=layer.get('in_features');outf=layer.get('out_features')
                if not isinstance(module,str) or module in modules or len(modules)>=2048 or any(type(x)!=int or not 0<x<=16384 for x in (inf,outf)):raise LoraError('Invalid or duplicate LoRA layer')
                if layer.get('capacity')!=capacity or layer.get('a_shape')!=[inf,capacity] or layer.get('b_shape')!=[capacity,outf]:raise LoraError('LoRA layer rank/shape mismatch')
                for letter in ('a','b'):
                    name=layer.get(letter)
                    if not re.fullmatch(r'rin_lora_\d{4}_'+letter,str(name)) or name in used_inputs:raise LoraError('Invalid LoRA graph input')
                    used_inputs.add(name)
                    if part.get('qnn_inputs',{}).get(name)!=layer[letter+'_shape']:raise LoraError('Compiled LoRA shape mismatch')
                modules[module]=(stage,pid,layer);flat.append(layer)
                for alias in layer.get('aliases',[])+[module]:
                    if not isinstance(alias,str) or alias in aliases and aliases[alias]!=module:raise LoraError('Ambiguous LoRA layer alias')
                    aliases[alias]=module
            if used_inputs!=set(part.get('qnn_inputs',{})) or used_outputs!=set(part.get('qnn_outputs',{})):raise LoraError('Compiled graph tensor coverage mismatch')
            for label in ('input','output'):
                for binding in part[label+'_bindings']:
                    if part['qnn_'+label+'s'][binding['name']]!=binding['qnn_shape']:raise LoraError('Compiler interface record mismatch')
            available.update(x['name'] for x in part['outputs'])
        if flat!=spec.get('layers') or not set(spec['original_outputs']).issubset(available):raise LoraError('Incomplete final graph outputs')
    return modules,aliases

def map_adapters(selections,folder,modules,aliases):
    if len(selections)>4:raise LoraError('At most four LoRAs within each layer rank capacity')
    if not folder.is_dir():raise LoraError('Place .safetensors in Download/sdxl_qnn/Lora/')
    import unicodedata
    files={};mapped={};evidence=[]
    for path in folder.iterdir():
        if path.suffix.lower()=='.safetensors':files.setdefault(unicodedata.normalize('NFC',path.stem),[]).append(path)
    for selection in selections:
        candidates=files.get(selection.name,[])
        if len(candidates)!=1:raise LoraError('LoRA file missing or ambiguous: '+selection.name)
        path=candidates[0]
        if path.is_symlink() or not path.is_file() or path.resolve().parent!=folder.resolve():raise LoraError('Invalid LoRA file path')
        weights=SafeWeights(path);family=weights.metadata.get('ss_base_model_version','').lower()
        if family and not any(k in family for k in ('sdxl','illustrious')):raise LoraError('Adapter declares a different model: '+family)
        seen=set();pairs=weights.pairs()
        for pair in pairs:
            module=aliases.get(pair['alias'])
            if module is None:raise LoraError('Unsupported LoRA layer, not ignored: '+pair['alias']+'; component covers UNet linear layers only')
            if module in seen:raise LoraError('Duplicate aliases for one LoRA layer')
            seen.add(module);_,_,layer=modules[module]
            if (pair['in'],pair['out'])!=(layer['in_features'],layer['out_features']):raise LoraError('LoRA/base shape mismatch: '+module)
            entries=mapped.setdefault(module,[])
            if pair['rank']+sum(x[1]['rank'] for x in entries)>layer['capacity']:raise LoraError('Combined LoRA rank exceeds 64: '+module)
            entries.append((weights,pair,selection.weight))
        file_sha=digest(path);weights.check_unchanged()
        evidence.append({'name':selection.name,'weight':selection.weight,'sha256':file_sha,'bytes':path.stat().st_size,'modules':len(pairs)})
    return mapped,evidence

def verify_contexts(template,manifest,cache,stages):
    files={}
    for stage in stages:
        for part in manifest['graphs'][stage]['parts']:
            file=inside(template,part['context_file'])
            if not file.is_file() or file.stat().st_size!=part['context_bytes']:raise LoraError('Incomplete LoRA component: '+part['id'])
            stamp={'path':str(file),'stamp':_stamp(file),'sha256':part['context_sha256']};marker=cache/('verified_'+part['id']+'.json')
            try:valid=read_json(marker)==stamp
            except (OSError,ValueError):valid=False
            if not valid:
                if digest(file)!=part['context_sha256'] or _stamp(file)!=stamp['stamp']:raise LoraError('LoRA context integrity mismatch: '+part['id'])
                atomic_json(marker,stamp)
            files[part['id']]=str(file)
    return files

def prepare(clean,negative,base,width,height,selections,log=print):
    base=Path(base).resolve();template=inside(base,RELATIVE);mp=template/'lora_template.json'
    if not mp.is_file():raise LoraError('LoRA 生图组件未安装，请在 LoRA 管理页安装；原底模无需重下。')
    manifest=read_json(mp);modules,aliases=validate(manifest)
    if manifest['resolution']!=[width,height]:raise LoraError('LoRA component supports native 1024x1024 only')
    mapped,evidence=map_adapters(selections,inside(base,'Lora'),modules,aliases)
    stages=[s for s,spec in manifest['graphs'].items() if any(x['module'] in mapped for x in spec['layers'])]
    if not stages:raise LoraError('No matching LoRA modules')
    cache=inside(base,'.rin_lora/partitioned');cache.mkdir(parents=True,exist_ok=True)
    contexts=verify_contexts(template,manifest,cache,stages)
    recipe={'format':FORMAT,'manifest':digest(mp),'adapters':evidence}
    key=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest();pointer=cache/(key+'.json');dest=None;banks=None
    expected={p['id']:{x[k]:math.prod(x[k+'_shape'])*4 for x in p['layers'] for k in ('a','b')} for s in stages for p in manifest['graphs'][s]['parts']}
    try:
        directory=read_json(pointer)['directory']
        if not re.fullmatch(re.escape(key)+r'_[0-9a-f]{32}',directory):raise LoraError('Invalid cache directory')
        candidate=inside(cache,directory);saved=read_json(candidate/'ready.json');records=saved['banks']
        valid=saved.get('recipe')==recipe and isinstance(records,dict) and set(records)==set(expected);checked={}
        if valid:
            for pid,items in records.items():
                if set(items)!=set(expected[pid]):valid=False;break
                for name,item in items.items():
                    f=inside(candidate,item['file'])
                    if item['bytes']!=expected[pid][name] or not f.is_file() or f.stat().st_size!=item['bytes']:valid=False;break
                    if str(f) not in checked:checked[str(f)]=digest(f)
                    if checked[str(f)]!=item['sha256']:valid=False;break
        if valid:dest=candidate;banks=records
    except (OSError,ValueError,KeyError,TypeError,AttributeError):pass
    if banks is None:
        required=sum(sum(x.values()) for x in expected.values())
        if shutil.disk_usage(cache).free<required+256*1024*1024:raise LoraError('Not enough space for LoRA matrix cache')
        dest=cache/(key+'_'+uuid.uuid4().hex);dest.mkdir();banks={};zero={}
        for stage in stages:
            for part in manifest['graphs'][stage]['parts']:
                pid=part['id'];banks[pid]={}
                for layer in part['layers']:
                    i,o,cap=layer['in_features'],layer['out_features'],layer['capacity']
                    A=np.zeros((i,cap),np.float32);B=np.zeros((cap,o),np.float32);offset=0;entries=mapped.get(layer['module'],[])
                    for source,pair,strength in entries:
                        rank=pair['rank'];alpha=float(source.read(pair['alpha']).reshape(-1)[0]) if pair['alpha'] else float(rank)
                        if not math.isfinite(alpha) or alpha<0:raise LoraError('Invalid training alpha')
                        A[:,offset:offset+rank]=source.read(pair['down']).reshape(rank,i).T
                        B[offset:offset+rank,:]=source.read(pair['up']).reshape(o,rank).T*(strength*alpha/rank)
                        offset+=rank;source.check_unchanged()
                    for letter,array in [('a',A),('b',B)]:
                        if not np.isfinite(array).all():raise LoraError('Weighted LoRA overflow')
                        name=layer[letter];filename=pid+'_'+name+'.raw'
                        if not entries:filename=zero.setdefault(tuple(array.shape),'zero_'+'x'.join(map(str,array.shape))+'.raw')
                        f=dest/filename
                        if not f.exists():array.astype('<f4',copy=False).tofile(f)
                        banks[pid][name]={'file':filename,'bytes':array.nbytes,'sha256':digest(f)}
                log('[LoRA] prepared '+pid+' ('+str(len(part['layers']))+' layers)')
        atomic_json(dest/'ready.json',{'recipe':recipe,'banks':banks});atomic_json(pointer,{'directory':dest.name})
    for entries in mapped.values():
        for source,_,_ in entries:source.check_unchanged()
    paths={pid:{name:str(inside(dest,item['file'])) for name,item in entries.items()} for pid,entries in banks.items()}
    metadata={'active':True,'format':FORMAT,'model_id':MODEL_ID,'files':evidence,'mapped_modules':len(mapped),'rank_capacity':64,'partition_count':sum(len(manifest['graphs'][s]['parts']) for s in stages),'cache_dir':str(dest),'base_contexts_used_for':[s for s in manifest['graphs'] if s not in stages]}
    log('[LoRA] real adapter matrices ready: '+str(len(mapped))+' layers')
    return PartitionPlan(clean,negative,{s:contexts[manifest['graphs'][s]['parts'][0]['id']] for s in stages},{},metadata,manifest,contexts,paths)

@dataclass
class PartitionPlan(Plan):
    manifest:dict
    part_contexts:dict
    banks:dict
    def stage_for(self,context):
        return next((s for s,path in self.contexts.items() if Path(path).resolve()==Path(context).resolve()),None)
    def execute(self,stage,input_list,output_dir,single,*,work_dir,native=False,native_input=False,**kwargs):
        spec=self.manifest['graphs'][stage];out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
        for line in Path(input_list).read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.lstrip().startswith(('#','%')):continue
            row={}
            for token in line.split():
                if ':=' not in token:raise LoraError('Partitioned WAI requires named inputs')
                name,path=token.split(':=',1)
                if name in row or not Path(path).is_file():raise LoraError('Duplicate or missing graph input')
                row[name]=Path(path)
            if set(row)!=set(spec['original_inputs']):raise LoraError('Original input list disagrees with partition manifest')
            rows.append(row)
        if not 1<=len(rows)<=4:raise LoraError('Invalid partition input row count')
        total_ms=0.0;profile=kwargs.pop('profile_tag',stage);kwargs.pop('model_path',None)
        prepare_output_dir(out,work_dir)
        with tempfile.TemporaryDirectory(prefix='_lora_parts_',dir=out) as temp_name:
            temp=Path(temp_name);states=[]
            shapes={x['logical']:x['shape'] for p in spec['parts'] for x in p['input_bindings']}
            for rid,row in enumerate(rows):
                state={}
                for name,path in row.items():
                    shape=shapes[name];size=math.prod(shape)
                    if path.stat().st_size!=size*(2 if native_input else 4):raise LoraError('Partition input shape/bytes mismatch: '+name)
                    if native_input:
                        array=np.fromfile(path,dtype='<f2').astype(np.float32)
                        if not np.isfinite(array).all():raise LoraError('Nonfinite partition input')
                        path=temp/(str(rid)+'_'+name+'.raw');array.tofile(path)
                    state[name]=(path,shape)
                states.append(state)
            for part in spec['parts']:
                pid=part['id'];part_out=temp/pid;part_rows=[]
                for rid,state in enumerate(states):
                    entries=[]
                    for binding in part['input_bindings']:
                        path,shape=state[binding['logical']]
                        if shape!=binding['shape']:raise LoraError('Partition state shape mismatch')
                        axes=binding['to_qnn_axes']
                        if axes!=list(range(len(shape))):
                            array=np.fromfile(path,'<f4').reshape(shape).transpose(axes)
                            converted=temp/(pid+'_'+str(rid)+'_'+binding['name']+'.raw');np.ascontiguousarray(array).tofile(converted);path=converted
                        entries.append(named_input(binding['name'],path))
                    entries.extend(named_input(name,path) for name,path in self.banks[pid].items());part_rows.append(entries)
                listing=temp/(pid+'.inputs.txt');write_input_rows(listing,part_rows)
                elapsed=single(self.part_contexts[pid],str(listing),str(part_out),native=False,native_input=False,profile_tag=str(profile)+'_'+pid,**kwargs);total_ms+=float(elapsed)
                for rid,state in enumerate(states):
                    for binding in part['output_bindings']:
                        path,info=resolve_output(part_out/('Result_'+str(rid)),binding['name'],expected_elements=math.prod(binding['qnn_shape']))
                        if info.get('shape')!=binding['qnn_shape'] or info['dtype']!='float32':raise LoraError('Actual partition output layout disagrees with compiler metadata')
                        axes=binding['to_qnn_axes'];shape=binding['shape']
                        if axes!=list(range(len(shape))):
                            array=np.fromfile(path,'<f4').reshape(binding['qnn_shape']).transpose(np.argsort(axes))
                            converted=temp/(pid+'_'+str(rid)+'_'+binding['logical']+'_logical.raw');np.ascontiguousarray(array).tofile(converted);path=converted
                        state[binding['logical']]=(path,shape)
            for rid,state in enumerate(states):
                result=out/('Result_'+str(rid));result.mkdir();records=[]
                for name in spec['original_outputs']:
                    path,shape=state[name];dest=result/(name+'.raw');shutil.copyfile(path,dest)
                    records.append({'name':name,'file':dest.name,'dtype':'float32','shape':shape,'bytes':dest.stat().st_size})
                    if native:
                        data=np.fromfile(dest,'<f4').astype('<f2');nf=result/(name+'_native.raw');data.tofile(nf)
                        records.append({'name':name,'file':nf.name,'dtype':'float16','shape':shape,'bytes':nf.stat().st_size})
                atomic_json(result/MANIFEST_NAME,{'schema':1,'complete':True,'outputs':records})
        return total_ms
