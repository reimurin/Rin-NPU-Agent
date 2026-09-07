"""alpha.4 unified WAI UNet routing.

Keeps the alpha.3 partition executor intact and adds one reversible activation
contract: when activation.json says active, ordinary generation, LoRA weight 0,
and non-zero LoRA all execute the same seven-part rank64 WAI UNet.  The legacy
two-file UNet remains a rollback point until handset quality/performance passes.
"""
from pathlib import Path
import hashlib,json,math,shutil,uuid
import numpy as np
import rin_lora as legacy
import rin_lora_partitioned as part
from rin_lora import LoraError,Plan,digest,_stamp

ACTIVATION_FILE='activation.json'

def _component(base):
    return part.inside(Path(base).resolve(),part.RELATIVE)

def activation_path(base):
    return _component(base)/ACTIVATION_FILE

def active(base):
    try:
        marker=activation_path(base)
        if not marker.is_file() or marker.stat().st_size>64*1024:return False
        data=part.read_json(marker,64*1024)
        return data.get('schema')==1 and data.get('model_id')==part.MODEL_ID and data.get('active') is True
    except (OSError,ValueError,KeyError,TypeError,LoraError):
        return False

def set_active(base,enabled,reason=''):
    marker=activation_path(base);marker.parent.mkdir(parents=True,exist_ok=True)
    part.atomic_json(marker,{'schema':1,'model_id':part.MODEL_ID,'active':bool(enabled),'reason':str(reason)[:240]})

def component_ready(base):
    try:
        template=_component(base);mp=template/'lora_template.json'
        if not mp.is_file():return False
        manifest=part.read_json(mp);part.validate(manifest)
        for stage in ('encoder','decoder'):
            for spec in manifest['graphs'][stage]['parts']:
                file=part.inside(template,spec['context_file'])
                if not file.is_file() or file.stat().st_size!=spec['context_bytes']:return False
        return True
    except (OSError,ValueError,KeyError,TypeError,LoraError):
        return False

def _prepare_unified(clean,negative,base,width,height,selections,log):
    base=Path(base).resolve();template=_component(base);mp=template/'lora_template.json'
    if not mp.is_file():raise LoraError('统一 WAI 模型未安装完整；旧普通 UNet 仍可回退。')
    manifest=part.read_json(mp);modules,aliases=part.validate(manifest)
    if manifest['resolution']!=[width,height]:raise LoraError('统一 WAI 模型当前只支持原生 1024x1024')
    mapped,evidence=part.map_adapters(selections,part.inside(base,'Lora'),modules,aliases) if selections else ({},[])
    stages=['encoder','decoder']
    cache=part.inside(base,'.rin_lora/unified');cache.mkdir(parents=True,exist_ok=True)
    contexts=part.verify_contexts(template,manifest,cache,stages)
    recipe={'format':'rin-wai-unified-alpha4-v1','manifest':digest(mp),'adapters':evidence}
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
    metadata={'active':bool(evidence),'unified_model':True,'zero_delta':not bool(evidence),'format':'rin-wai-unified-alpha4-v1','model_id':part.MODEL_ID,'files':evidence,'mapped_modules':len(mapped),'rank_capacity':64,'partition_count':7,'cache_dir':str(dest),'base_contexts_used_for':[]}
    return part.PartitionPlan(clean,negative,{s:contexts[manifest['graphs'][s]['parts'][0]['id']] for s in stages},{},metadata,manifest,contexts,paths)

def prepare(prompt,negative,base,width,height,log=print):
    clean,selections=legacy.parse_tags(prompt);neg,negative_tags=legacy.parse_tags(negative)
    if negative_tags:raise LoraError('Place LoRA controls in the positive prompt, not the negative prompt')
    enabled=[s for s in selections if s.weight!=0]
    if not active(base):
        return legacy.prepare(prompt,negative,base,width,height,log=log)
    try:
        plan=_prepare_unified(clean,neg,base,width,height,enabled,log)
        log('[Unified WAI] active: seven-part UNet; '+('LoRA matrices applied' if enabled else 'zero-delta baseline'))
        return plan
    except Exception as e:
        try:set_active(base,False,'runtime_validation_failed')
        except Exception:pass
        if enabled:raise
        log('[Unified WAI] validation failed; automatic rollback to legacy UNet: '+str(e))
        return Plan(clean,neg,{}, {},{'active':False,'unified_model':False,'rollback':True,'rollback_reason':str(e)[:500]})
