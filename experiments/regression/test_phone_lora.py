"""Checks use only new project-local fixtures, never real QNN execution."""
from pathlib import Path
import argparse,sys,json,struct,tempfile
import numpy as np
from safetensors.numpy import save_file
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/main/assets/sdxl_runtime'))
import phone_lora as l

def main(out):
    out=Path(out)
    if out.exists():raise RuntimeError('Use a new result directory')
    out.mkdir(parents=True);checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try:fn()
        except (l.LoraError,ValueError):check(name,True)
        else:raise AssertionError(name)
    base=out/'fixture';folder=base/'Lora';folder.mkdir(parents=True)
    manifest=base/l.TEMPLATE_REL;manifest.parent.mkdir(parents=True)
    spec={'schema':1,'complete':True,'model_id':'synthetic-fixture','rank_capacity':16,'resolution':[1024,1024],'graphs':{}}
    for graph in ['encoder','decoder']:
        ctx=manifest.parent/(graph+'.bin');ctx.write_bytes(('test context '+graph).encode())
        module='unet.'+graph+'.to_q'
        layer={'module':module,'aliases':[module,'lora_'+module.replace('.','_')],'in_features':8,'out_features':6,'a':'rin_lora_0000_a','b':'rin_lora_0000_b','a_shape':[8,16],'b_shape':[16,6],'capacity':16}
        spec['graphs'][graph]={'layers':[layer],'context_file':ctx.name,'context_bytes':ctx.stat().st_size,'context_sha256':l.sha(ctx)}
    manifest.write_text(json.dumps(spec))
    a=np.arange(24,dtype=np.float32).reshape(3,8)/30
    b=np.arange(18,dtype=np.float32).reshape(6,3)/20
    def adapter(name,style='kohya',rank=3,alpha=2.,dtype=np.float32,extra=None):
        prefix='lora_unet_encoder_to_q' if style=='kohya' else 'unet.encoder.to_q'
        down='.lora_down.weight' if style=='kohya' else '.lora_A.weight'
        up='.lora_up.weight' if style=='kohya' else '.lora_B.weight'
        aa=a if rank==3 else np.ones((rank,8),np.float32)
        bb=b if rank==3 else np.ones((6,rank),np.float32)
        tensors={prefix+down:aa.astype(dtype),prefix+up:bb.astype(dtype)}
        if alpha is not None:tensors[prefix+'.alpha']=np.asarray(alpha,np.float32)
        if extra:tensors.update(extra)
        target=folder/(name+'.safetensors')
        if target.exists():raise RuntimeError('Fixture already exists')
        save_file(tensors,str(target),metadata={'ss_base_model_version':'sdxl_base_v1-0'})
        return target
    def prepare(prompt,negative=''):
        return l.prepare_request(prompt,negative,base,log=lambda _:None)
    def matrices(session,graph='encoder'):
        entries=session.entries[graph]
        return np.fromfile(entries['rin_lora_0000_a'],np.float32).reshape(8,16),np.fromfile(entries['rin_lora_0000_b'],np.float32).reshape(16,6)
    check('plain_prompt_baseline',prepare('cat','blur')==('cat','blur',None))
    clean,negative,session=prepare('cat <lora:missing:0>')
    check('zero_strength_no_file_required',session is None and 'lora' not in clean)
    adapter('角色')
    for strength in [0.8,1.1,-0.5]:
        _,_,session=prepare('<lora:角色:'+str(strength)+'>')
        aa,bb=matrices(session)
        check('strength_'+str(strength),np.allclose(aa@bb,(a.T@b.T)*(strength*2/3),rtol=1e-6,atol=1e-6))
    aa,bb=matrices(session,'decoder');check('untouched_graph_zero',np.all(aa==0) and np.all(bb==0))
    _,_,cached=prepare('<lora:角色:-0.5>');check('cache_reused',cached.entries==session.entries)
    adapter('second')
    _,_,mixed=prepare('<lora:角色:.8><lora:second:1.1>');aa,bb=matrices(mixed)
    check('combined_independent_strength',np.allclose(aa@bb,(a.T@b.T)*(1.9*2/3),rtol=1e-6,atol=1e-6))
    adapter('peft',style='peft');_,_,peft=prepare('<lora:peft:.8>');check('peft_pair_read',peft.report['matched_modules']==1)
    adapter('half',dtype=np.float16);_,_,half=prepare('<lora:half:.8>');aa,bb=matrices(half)
    check('half_precision_values',np.allclose(aa@bb,(a.astype(np.float16).astype(np.float32).T@b.astype(np.float16).astype(np.float32).T)*(.8*2/3),rtol=1e-6,atol=1e-6))
    adapter('default_alpha',alpha=None);_,_,default=prepare('<lora:default_alpha:.8>');aa,bb=matrices(default)
    check('missing_training_alpha',np.allclose(aa@bb,(a.T@b.T)*.8,rtol=1e-6,atol=1e-6))
    adapter('rank12a',rank=12);adapter('rank12b',rank=12)
    rejects('combined_rank_limit',lambda:prepare('<lora:rank12a:1><lora:rank12b:1>'))
    adapter('unsupported',extra={'other.hada_w1_a':np.ones((2,2),np.float32)})
    rejects('extra_weights_not_ignored',lambda:prepare('<lora:unsupported:1>'))
    adapter('te',extra={'lora_te1_layer.lora_down.weight':np.ones((3,8),np.float32),'lora_te1_layer.lora_up.weight':np.ones((6,3),np.float32)})
    rejects('text_encoder_not_ignored',lambda:prepare('<lora:te:1>'))
    rejects('missing_adapter',lambda:prepare('<lora:not-found:1>'))
    rejects('conflict_between_prompts',lambda:prepare('<lora:角色:1>','<lora:角色:.8>'))
    for i,prompt in enumerate(['<lora:foo:nan>','<lora:../foo:1>','<lora:foo:3>','<lora:foo:>','<lora:foo:inf>']):rejects('invalid_tag_'+str(i),lambda prompt=prompt:prepare(prompt))
    path=out/'input_list.txt';path.write_text('sample:=one.raw\nsample:=two.raw\n')
    session.extend_input_list(session.contexts['encoder'],path);first=path.read_text();session.extend_input_list(session.contexts['encoder'],path)
    check('two_cfg_rows_idempotent',path.read_text()==first and first.count('rin_lora_0000_a:=')==2)
    clip=out/'clip.txt';clip.write_text('ids.raw\n');session.extend_input_list('clip.bin',clip);check('clip_inputs_unchanged',clip.read_text()=='ids.raw\n')
    bad=out/'bad.safetensors';bad.write_bytes(struct.pack('<Q',2)+b'[]');rejects('invalid_header_map',lambda:l.SafeWeights(bad))
    raw=folder/'角色.safetensors';truncated=out/'short.safetensors';truncated.write_bytes(raw.read_bytes()[:-1]);rejects('truncated_payload',lambda:l.SafeWeights(truncated))
    check('zero_again_returns_baseline',prepare('<lora:角色:0>')[2] is None)
    report={'tests':len(checks),'passed':True,'checks':checks,'scope':'actual phone LoRA module, local fixture contexts, numerical validation, no NPU inference'}
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();main(args.out)
