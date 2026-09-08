"""alpha.5 unified WAI UNet routing.

Ordinary generation, zero-weight LoRA and active LoRA use the same seven-part
rank64 WAI UNet whenever a resolution-specific component is installed.  The
existing 1024x1024 component keeps its historical model id/path.  Additional
native resolutions use their own model id, activation marker and matrix cache.
"""
from pathlib import Path
import hashlib,json,math,shutil,uuid,re
import numpy as np
import rin_lora as legacy
import rin_lora_partitioned as part
from rin_lora import LoraError,Plan,digest

ACTIVATION_FILE='activation.json'
LEGACY_MEMORY_ROLLBACK_PREFIX='alpha4_persistent_memory_rollback'

SHARED_PACK_ROOT='context/model_packs'
SHARED_PACK_SCHEMA=1
SHARED_RUNTIME_ABI=1


def _shared_pack_candidates(base):
    root=part.inside(Path(base).resolve(),SHARED_PACK_ROOT)
    if not root.is_dir():return []
    folders=[];pointer=root/'current.json';current_selected=False
    try:
        if pointer.is_file() and pointer.stat().st_size<64*1024:
            current=part.read_json(pointer,64*1024).get('directory','')
            if isinstance(current,str) and current and '..' not in Path(current).parts:
                chosen=part.inside(root,current)
                if chosen.is_dir():folders.append(chosen);current_selected=True
    except (OSError,ValueError,TypeError,KeyError,LoraError):pass
    if not current_selected:return []
    out=[]
    for folder in folders:
        mp=folder/'model_manifest.json'
        try:
            if not mp.is_file() or mp.stat().st_size>2*1024*1024:continue
            data=part.read_json(mp,2*1024*1024)
            if data.get('schema')!=SHARED_PACK_SCHEMA or data.get('complete') is not True or data.get('runtime_abi')!=SHARED_RUNTIME_ABI:continue
            target=data.get('target',{})
            if target.get('qnn_soc_id')!=69 or target.get('dsp_arch')!=79:continue
            out.append((folder,data))
        except (OSError,ValueError,TypeError,KeyError,LoraError):continue
    return out


def _pack_file_ok(folder,item):
    if not isinstance(item,dict):return False
    rel=item.get('file','');expected=item.get('bytes',0)
    if not isinstance(rel,str) or not rel or '..' in Path(rel).parts:return False
    f=part.inside(folder,rel)
    return f.is_file() and f.stat().st_size>0 and (not expected or f.stat().st_size==expected)


def _shared_resolution_entry(data,width,height):
    for item in data.get('supported_resolutions',data.get('resolutions',[])):
        if isinstance(item,dict) and item.get('width')==int(width) and item.get('height')==int(height):return item
    return None


def shared_runtime(base,width,height):
    for folder,data in _shared_pack_candidates(base):
        entry=_shared_resolution_entry(data,width,height)
        if not entry:continue
        contexts=data.get('contexts',{});vae=data.get('vae',{})
        keys=('encoder_p0','encoder_p1','encoder_p2','decoder_p0','decoder_p1','decoder_p2','decoder_p3')
        try:
            if any(not _pack_file_ok(folder,contexts.get(k)) for k in keys) or not _pack_file_ok(folder,vae):continue
            graph=str(entry.get('graph',''));template=part.inside(folder,str(entry.get('template','')))
            if not re.fullmatch(r'_[0-9]+x[0-9]+',graph) or not template.is_file():continue
            paths={k:str(part.inside(folder,contexts[k]['file'])) for k in keys}
            return {'pack_dir':str(folder),'pack_id':str(data.get('pack_id',data.get('id',folder.name))),'version':str(data.get('model_pack_version',data.get('version',''))),
                    'template':str(template),'context_root':str(folder.resolve() if str(data.get('contexts_root','contexts')).strip() in ('.','./') else part.inside(folder,str(data.get('contexts_root','contexts')).strip() or 'contexts')),'contexts':paths,
                    'encoder':paths['encoder_p0'],'decoder':paths['decoder_p0'],'vae':str(part.inside(folder,vae['file'])),'graph':graph,
                    'lora_abi':str(data.get('lora',{}).get('abi_signature',''))}
        except (OSError,ValueError,TypeError,KeyError,LoraError):continue
    return None


def shared_resolutions(base):
    found=set()
    for _,data in _shared_pack_candidates(base):
        for item in data.get('supported_resolutions',data.get('resolutions',[])):
            if not isinstance(item,dict):continue
            w,h=item.get('width'),item.get('height')
            if type(w) is int and type(h) is int and shared_runtime(base,w,h):found.add((w,h))
    return sorted(found,key=lambda r:(r[0]*r[1],r[0]))


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
        if shared_runtime(base,width,height):return True
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
    base=Path(base).resolve();shared=shared_runtime(base,width,height)
    if shared:
        mp=Path(shared['template']).resolve();template=mp.parent;context_root=Path(shared['context_root']).resolve()
    else:
        template=_component(base,width,height);mp=template/'lora_template.json';context_root=template
    if not mp.is_file():raise LoraError(f'统一 WAI 模型未安装完整：{width}x{height}')
    manifest=part.read_json(mp);modules,aliases=_validate_manifest(manifest,width,height)
    available_parts={spec['id'] for stage in ('encoder','decoder') for spec in manifest['graphs'][stage]['parts']}
    requested=set(persistent_parts)
    if len(requested)!=len(persistent_parts) or not requested.issubset(available_parts):
        unknown=sorted(requested-available_parts)
        raise LoraError('Unsupported persistent graph part: '+(', '.join(unknown) if unknown else 'duplicate id'))
    mapped,evidence=part.map_adapters(selections,part.inside(base,'Lora'),modules,aliases) if selections else ({},[])
    stages=['encoder','decoder']
    abi=shared['lora_abi'] if shared else '';cache_key=('shared_'+abi) if abi else f'{width}x{height}';cache=part.inside(base,f'.rin_lora/unified/{cache_key}');cache.mkdir(parents=True,exist_ok=True)
    contexts=part.verify_contexts(template,manifest,cache,stages,context_root=context_root)
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
    metadata={'active':bool(evidence),'unified_model':True,'zero_delta':not bool(evidence),'format':'rin-wai-unified-alpha5-v1','model_id':model_id_for(width,height),'resolution':[width,height],'shared_model_pack':bool(shared),'graph_name':shared['graph'] if shared else '','model_pack_id':shared['pack_id'] if shared else '','model_pack_version':shared['version'] if shared else '','lora_abi':abi,'files':evidence,'mapped_modules':len(mapped),'rank_capacity':64,'partition_count':sum(len(manifest['graphs'][s]['parts']) for s in stages),'cache_dir':str(dest),'base_contexts_used_for':[],'persistent_parts':sorted(requested)}
    return part.PartitionPlan(clean,negative,{s:contexts[manifest['graphs'][s]['parts'][0]['id']] for s in stages},{},metadata,manifest,contexts,paths,requested)


def prepare(prompt,negative,base,width,height,log=print):
    clean,selections=legacy.parse_tags(prompt);neg,negative_tags=legacy.parse_tags(negative)
    if negative_tags:raise LoraError('Place LoRA controls in the positive prompt, not the negative prompt')
    enabled=[s for s in selections if s.weight!=0]
    shared=shared_runtime(base,width,height)
    if shared:
        try:
            plan=_prepare_unified(clean,neg,base,width,height,enabled,(),log)
            log(f"[Unified WAI] alpha.6 shared pack {shared['pack_id']} {shared['version']} graph={shared['graph']}")
            return plan
        except Exception as e:
            if enabled:raise
            log('[Unified WAI] alpha.6 shared validation failed; rollback to legacy UNet: '+str(e))
            return Plan(clean,neg,{}, {},{'active':False,'unified_model':False,'shared_model_pack':True,'rollback':True,'rollback_reason':str(e)[:500],'resolution':[width,height]})
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
