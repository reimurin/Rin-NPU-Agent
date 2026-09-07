"""alpha.5 unified WAI UNet routing.

Ordinary generation, zero-weight LoRA and active LoRA use the same seven-part
rank64 WAI UNet whenever a resolution-specific component is installed.  The
existing 1024x1024 component keeps its historical model id/path.  Additional
native resolutions use their own model id, activation marker and matrix cache.
"""
from pathlib import Path
import hashlib,json,math,shutil,uuid
import numpy as np
import rin_lora as legacy
import rin_lora_partitioned as part
from rin_lora import LoraError,Plan,digest

ACTIVATION_FILE='activation.json'
LEGACY_MEMORY_ROLLBACK_PREFIX='alpha4_persistent_memory_rollback'


def model_id_for(width,height):
    width,height=int(width),int(height)
    if (width,height)==(1024,1024):return part.MODEL_ID
    if not 64<=width<=8192 or not 64<=height<=8192 or width%8 or height%8:
        raise LoraError('Invalid unified WAI resolution')
    return f'wai-v170-sm8750-lora-r64-{width}x{height}-partitioned-v1'


def relative_for(width,height):return 'context/lora/'+model_id_for(width,height)


def _component(base,width=1024,height=1024):
    return part.inside(Path(base).resolve(),relative_for(width,height))


def _validate_manifest(manifest,width,height):
    expected_id=model_id_for(width,height)
    if not isinstance(manifest,dict) or manifest.get('model_id')!=expected_id:
        raise LoraError('Unified component model id does not match requested resolution')
    if manifest.get('resolution')!=[int(width),int(height)]:
        raise LoraError('Unified component resolution metadata mismatch')
    normalized=json.loads(json.dumps(manifest))
    normalized['model_id']=part.MODEL_ID
    normalized['resolution']=[1024,1024]
    return part.validate(normalized)


def activation_path(base,width=1024,height=1024):
    return _component(base,width,height)/ACTIVATION_FILE


def activation_config(base,width=1024,height=1024):
    marker=activation_path(base,width,height)
    if not marker.is_file():return {'exists':False,'active':False,'reason':'','persistent_parts':[]}
    if marker.stat().st_size>64*1024:raise LoraError('Oversized unified activation metadata')
    data=part.read_json(marker,64*1024);expected_id=model_id_for(width,height)
    if data.get('schema')!=1 or data.get('model_id')!=expected_id:raise LoraError('Unified activation metadata does not match this model')
    raw=data.get('persistent_parts',[])
    if not isinstance(raw,list) or len(raw)>8 or any(not isinstance(x,str) or not x or len(x)>64 for x in raw):raise LoraError('Invalid persistent part list')
    if len(raw)!=len(set(raw)):raise LoraError('Duplicate persistent part id')
    return {'exists':True,'active':data.get('active') is True,'reason':str(data.get('reason',''))[:240],'persistent_parts':raw}


def active(base,width=1024,height=1024):
    try:return activation_config(base,width,height)['active']
    except (OSError,ValueError,KeyError,TypeError,LoraError):return False


def set_active(base,enabled,reason='',persistent_parts=(),width=1024,height=1024):
    values=list(persistent_parts)
    if len(values)>8 or any(not isinstance(x,str) or not x or len(x)>64 for x in values) or len(values)!=len(set(values)):raise LoraError('Invalid persistent part list')
    marker=activation_path(base,width,height);marker.parent.mkdir(parents=True,exist_ok=True)
    part.atomic_json(marker,{'schema':1,'model_id':model_id_for(width,height),'active':bool(enabled),'reason':str(reason)[:240],'persistent_parts':values})


def component_ready(base,width=1024,height=1024):
    try:
        template=_component(base,width,height);mp=template/'lora_template.json'
        if not mp.is_file():return False
        manifest=part.read_json(mp);_validate_manifest(manifest,width,height)
        for stage in ('encoder','decoder'):
            for spec in manifest['graphs'][stage]['parts']:
                file=part.inside(template,spec['context_file'])
                if not file.is_file() or file.stat().st_size!=spec['context_bytes']:return False
        return True
    except (OSError,ValueError,KeyError,TypeError,LoraError):
        return False


def _ensure_alpha5_activation(base,width,height,log):
    config=activation_config(base,width,height)
    if config['exists'] and config['active']:return config
    if not component_ready(base,width,height):return config
    if not config['exists']:
        set_active(base,True,'alpha5_unified_default',(),width,height)
        log(f'[Unified WAI] alpha.5 enabled installed {width}x{height} component with persistent contexts off')
        return activation_config(base,width,height)
    if config['reason'].startswith(LEGACY_MEMORY_ROLLBACK_PREFIX):
        set_active(base,True,'alpha5_migrated_safe_nonpersistent',(),width,height)
        log('[Unified WAI] alpha.4 persistent-memory rollback migrated to alpha.5 safe non-persistent unified mode')
        return activation_config(base,width,height)
    return config


def _prepare_unified(clean,negative,base,width,height,selections,persistent_parts,log):
    base=Path(base).resolve();template=_component(base,width,height);mp=template/'lora_template.json'
    if not mp.is_file():raise LoraError(f'统一 WAI 模型未安装完整：{width}x{height}')
    manifest=part.read_json(mp);modules,aliases=_validate_manifest(manifest,width,height)
    available_parts={spec['id'] for stage in ('encoder','decoder') for spec in manifest['graphs'][stage]['parts']}
    requested=set(persistent_parts)
    if len(requested)!=len(persistent_parts) or not requested.issubset(available_parts):
        unknown=sorted(requested-available_parts)
        raise LoraError('Unsupported persistent graph part: '+(', '.join(unknown) if unknown else 'duplicate id'))
    mapped,evidence=part.map_adapters(selections,part.inside(base,'Lora'),modules,aliases) if selections else ({},[])
    stages=['encoder','decoder']
    cache=part.inside(base,f'.rin_lora/unified/{width}x{height}');cache.mkdir(parents=True,exist_ok=True)
    contexts=part.verify_contexts(template,manifest,cache,stages)
    recipe={'format':'rin-wai-unified-alpha5-v1','resolution':[width,height],'manifest':digest(mp),'adapters':evidence}
    key=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest();pointer=cache/(key+'.json');dest=None;banks=None
    expected={p['id']:{x[k]:math.prod(x[k+'_shape'])*4 for x in p['layers'] for k in ('a','b')} for s in stages for p in manifest['graphs'][s]['parts']}
    try:
        directory=part.read_json(pointer)['directory']
        if not directory.startswith(key+'_'):raise LoraError('Invalid unified cache directory')
        candidate=part.inside(cache,directory);saved=part.read_json(candidate/'ready.json');records=saved['banks']
        valid=saved.get('recipe')==recipe and isinstance(records,dict) and set(records)==set(expected);checked={}
        if valid:
            for pid,items in records.items():
                if set(items)!=set(expected[pid]):valid=False;break
                for name,item in items.items():
                    f=part.inside(candidate,item['file'])
                    if item['bytes']!=expected[pid][name] or not f.is_file() or f.stat().st_size!=item['bytes']:valid=False;break
                    if str(f) not in checked:checked[str(f)]=digest(f)
                    if checked[str(f)]!=item['sha256']:valid=False;break
        if valid:dest=candidate;banks=records
    except (OSError,ValueError,KeyError,TypeError,AttributeError,LoraError):
        pass
    if banks is None:
        required=sum(sum(x.values()) for x in expected.values())
        if shutil.disk_usage(cache).free<required+256*1024*1024:raise LoraError('统一模型 LoRA/zero-delta 矩阵缓存空间不足')
        dest=cache/(key+'_'+uuid.uuid4().hex);dest.mkdir();banks={};zero={}
        for stage in stages:
            for spec in manifest['graphs'][stage]['parts']:
                pid=spec['id'];banks[pid]={}
                for layer in spec['layers']:
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
                log('[Unified WAI] prepared '+pid+' ('+str(len(spec['layers']))+' layers)')
        part.atomic_json(dest/'ready.json',{'recipe':recipe,'banks':banks});part.atomic_json(pointer,{'directory':dest.name})
    for entries in mapped.values():
        for source,_,_ in entries:source.check_unchanged()
    paths={pid:{name:str(part.inside(dest,item['file'])) for name,item in items.items()} for pid,items in banks.items()}
    metadata={'active':bool(evidence),'unified_model':True,'zero_delta':not bool(evidence),'format':'rin-wai-unified-alpha5-v1','model_id':model_id_for(width,height),'resolution':[width,height],'files':evidence,'mapped_modules':len(mapped),'rank_capacity':64,'partition_count':sum(len(manifest['graphs'][s]['parts']) for s in stages),'cache_dir':str(dest),'base_contexts_used_for':[],'persistent_parts':sorted(requested)}
    return part.PartitionPlan(clean,negative,{s:contexts[manifest['graphs'][s]['parts'][0]['id']] for s in stages},{},metadata,manifest,contexts,paths,requested)


def prepare(prompt,negative,base,width,height,log=print):
    clean,selections=legacy.parse_tags(prompt);neg,negative_tags=legacy.parse_tags(negative)
    if negative_tags:raise LoraError('Place LoRA controls in the positive prompt, not the negative prompt')
    enabled=[s for s in selections if s.weight!=0]
    try:
        config=_ensure_alpha5_activation(base,width,height,log)
        if not config['active']:
            if (width,height)!=(1024,1024) and component_ready(base,width,height):raise LoraError('Unified component is installed but disabled for this resolution')
            return legacy.prepare(prompt,negative,base,width,height,log=log)
        plan=_prepare_unified(clean,neg,base,width,height,enabled,config['persistent_parts'],log)
        suffix='; persistent='+(','.join(plan.metadata['persistent_parts']) if plan.metadata['persistent_parts'] else 'off')
        log(f'[Unified WAI] active {width}x{height}: seven-part UNet; '+('LoRA matrices applied' if enabled else 'zero-delta baseline')+suffix)
        return plan
    except Exception as e:
        try:set_active(base,False,'runtime_validation_failed',(),width,height)
        except Exception:pass
        if enabled:raise
        log('[Unified WAI] validation failed; automatic rollback to legacy UNet: '+str(e))
        return Plan(clean,neg,{}, {},{'active':False,'unified_model':False,'rollback':True,'rollback_reason':str(e)[:500],'resolution':[width,height]})
