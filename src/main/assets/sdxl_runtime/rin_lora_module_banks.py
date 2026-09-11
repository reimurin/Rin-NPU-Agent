"""Real safetensors adapters for V79 WAI contexts with per-layer LoRA inputs.
Uses exact concatenated low-rank matrices; no lossy merge or ignored modules.
"""
from pathlib import Path
import copy,json,math,os,re,shutil,uuid
import numpy as np
from rin_lora import LoraError,Plan,_inside,_json,_stamp,digest,map_weights,validate_layout

FORMAT='rin-lora-module-banks-v1'

def checked_manifest(manifest):
    if manifest.get('format')!=FORMAT or manifest.get('soc')!=69 or manifest.get('dsp')!=79:
        raise LoraError('LoRA component format or target device mismatch')
    packed=copy.deepcopy(manifest);packed['format']='rin-lora-packed-linear-v1'
    validate_layout(packed)
    for stage,spec in manifest['stages'].items():
        if spec.get('input_layout')!='per_module' or len(spec['modules'])>2000:
            raise LoraError('Invalid per-layer LoRA input layout')
        names=[]
        for index,module in enumerate(spec['modules']):
            name=module.get('input_name');count=module.get('input_elements')
            if name!='rin_lora_m'+str(index).zfill(4) or count!=module['a_elements']+module['b_elements']:
                raise LoraError('Invalid per-layer LoRA input name or size')
            if not 0<count*4<=4*1024*1024 or module['rank_capacity']!=manifest['rank_capacity']:
                raise LoraError('Per-layer LoRA input exceeds validated size or rank')
            names.append(name)
        if spec.get('input_names')!=names:raise LoraError('LoRA input list disagrees with layer map')
        if spec.get('context')!=stage+'.bin' or not isinstance(spec.get('context_bytes'),int) or spec['context_bytes']<1:
            raise LoraError('Invalid LoRA context file record')
        if not re.fullmatch('[0-9a-f]{64}',str(spec.get('context_sha256',''))):raise LoraError('Invalid context checksum')
    return packed

def record_name(stage,module):return stage+'/'+module['input_name']+'.raw'

def verify_records(dest,manifest,stages,records):
    if not isinstance(records,dict) or set(records)!=set(stages):return False
    for stage in stages:
        actual=records[stage]
        if not isinstance(actual,list) or len(actual)!=len(manifest['stages'][stage]['modules']):return False
        for entry,module in zip(actual,manifest['stages'][stage]['modules']):
            expected=record_name(stage,module)
            if not isinstance(entry,dict) or entry.get('file')!=expected or entry.get('input')!=module['input_name'] or entry.get('bytes')!=module['input_elements']*4:return False
            path=_inside(dest,expected)
            if not path.is_file() or path.stat().st_size!=entry['bytes'] or digest(path)!=entry.get('sha256'):return False
    return True

def pack_module_weights(manifest,mapped,out,stages,log):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);records={}
    for stage in stages:
        (out/stage).mkdir();records[stage]=[]
        for index,module in enumerate(manifest['stages'][stage]['modules']):
            size=module['input_elements'];buffer=np.zeros(size,dtype='<f4')
            inf,outf,capacity=module['in_features'],module['out_features'],module['rank_capacity']
            A=buffer[:module['a_elements']].reshape(inf,capacity)
            B=buffer[module['a_elements']:].reshape(capacity,outf)
            offset=0
            for source,pair,strength in mapped.get(module['module'],[]):
                rank=pair['rank'];down=source.read(pair['down']).reshape(rank,inf);up=source.read(pair['up']).reshape(outf,rank)
                alpha=float(source.read(pair['alpha']).reshape(-1)[0]) if pair['alpha'] else float(rank)
                if not math.isfinite(alpha) or alpha<0:raise LoraError('Invalid LoRA training alpha')
                A[:,offset:offset+rank]=down.T
                B[offset:offset+rank,:]=up.T*(strength*alpha/rank)
                offset+=rank;source.check_unchanged()
            if not np.isfinite(buffer).all():raise LoraError('LoRA weighted matrices overflow')
            name=record_name(stage,module);path=out/name
            with path.open('wb') as stream:buffer.tofile(stream)
            records[stage].append({'file':name,'input':module['input_name'],'bytes':size*4,'sha256':digest(path)})
            if index%80==0:log('[LoRA] '+stage+' matrices '+str(index+1)+'/'+str(len(manifest['stages'][stage]['modules'])))
    return records

def prepare_modules(clean,negative,base,manifest,raw,selections,folder,template,log=print):
    packed=checked_manifest(manifest)
    if not folder.is_dir():raise LoraError('Place compatible .safetensors LoRA files in Download/sdxl_qnn/Lora/')
    if len(selections)>4:raise LoraError('At most four LoRAs are accepted, within each layer rank capacity')
    mapped,evidence=map_weights(packed,selections,folder)
    for entries in mapped.values():
        for weights,_,_ in entries:
            family=weights.metadata.get('ss_base_model_version','').lower()
            if family and not ('sdxl' in family or 'illustrious' in family):raise LoraError('Adapter declares a different base model: '+family)
    stages=[stage for stage,spec in manifest['stages'].items() if any(module['module'] in mapped for module in spec['modules'])]
    if not stages:raise LoraError('No WAI LoRA modules selected')
    cache=_inside(base,'.rin_lora/cache');cache.mkdir(parents=True,exist_ok=True)
    contexts={}
    for stage in stages:
        spec=manifest['stages'][stage];path=_inside(template,spec['context'])
        if not path.is_file() or path.stat().st_size!=spec['context_bytes']:raise LoraError('LoRA context is not fully installed: '+stage)
        stamp={'path':str(path),'stamp':_stamp(path),'sha256':spec['context_sha256']}
        marker=cache/('verified_module_'+stage+'.json');valid=False
        try:valid=_json(marker.read_text())==stamp
        except (OSError,ValueError):pass
        if not valid:
            log('[LoRA] verifying '+stage+' context checksum')
            if digest(path)!=spec['context_sha256'] or _stamp(path)!=stamp['stamp']:raise LoraError('LoRA context changed or checksum mismatch: '+stage)
            pending=marker.with_name(marker.name+'.'+uuid.uuid4().hex+'.pending');pending.write_text(json.dumps(stamp));os.replace(pending,marker)
        contexts[stage]=str(path)
    recipe={'protocol':FORMAT,'manifest_sha256':__import__('hashlib').sha256(raw).hexdigest(),'files':evidence}
    key=__import__('hashlib').sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest();pointer=cache/(key+'.pointer.json');dest=None;records=None
    try:
        name=_json(pointer.read_text())['directory']
        if not isinstance(name,str) or not re.fullmatch(re.escape(key)+r'_[0-9a-f]{32}',name):raise LoraError('Invalid cache pointer')
        candidate=_inside(cache,name);saved=_json((candidate/'ready.json').read_text())
        if saved.get('recipe')==recipe and verify_records(candidate,manifest,stages,saved.get('packs')):
            dest=candidate;records=saved['packs']
    except (OSError,ValueError,KeyError,TypeError):pass
    if records is None:
        total=sum(manifest['stages'][stage]['elements']*4 for stage in stages)
        if shutil.disk_usage(cache).free<total+256*1024*1024:raise LoraError('Insufficient space for LoRA matrix cache')
        dest=cache/(key+'_'+uuid.uuid4().hex);records=pack_module_weights(manifest,mapped,dest,stages,log)
        if not verify_records(dest,manifest,stages,records):raise LoraError('LoRA cache verification failed')
        (dest/'ready.json').write_text(json.dumps({'recipe':recipe,'packs':records},indent=2),encoding='utf-8')
        pending=pointer.with_name(pointer.name+'.'+uuid.uuid4().hex+'.pending');pending.write_text(json.dumps({'directory':dest.name}));os.replace(pending,pointer)
    for entries in mapped.values():
        for source,_,_ in entries:source.check_unchanged()
    inputs={stage:[entry['input']+':='+str(dest/entry['file']) for entry in entries] for stage,entries in records.items()}
    metadata={'active':True,'format':FORMAT,'template_id':manifest['template_id'],'files':evidence,'mapped_modules':len(mapped),'rank_capacity':manifest['rank_capacity'],'cache_dir':str(dest),'input_counts':{s:len(v) for s,v in inputs.items()},'base_contexts_used_for':[s for s in manifest['stages'] if s not in contexts]}
    log('[LoRA] real adapters mapped: '+str(len(mapped))+' modules; per-layer banks ready')
    return Plan(clean,negative,contexts,inputs,metadata)
