"""Isolated CPU-only checks; creates only new tiny fixtures under --out-dir."""
import argparse,json,math,struct
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from prototype import LoraError,parse_prompt,runtime_paths,inspect_lora,scan_loras,lora_delta

def run(out):
    out=Path(out)
    if out.exists(): raise RuntimeError('Choose a new test output directory')
    out.mkdir(parents=True)
    results=[]
    def check(name,condition):
        passed=bool(condition);results.append({'name':name,'passed':passed})
        print(('PASS ' if passed else 'FAIL ')+name)
    def reject(name,func):
        try:func()
        except (LoraError,ValueError,OSError):check(name,True)
        else:check(name,False)
    for strength in [0,.8,1.1,-.5]:
        check('adapter_strength_'+str(strength),parse_prompt(f'<lora:角色:{strength}>').loras[0].weight==strength)
    check('default_strength',parse_prompt('<lora:角色>').loras[0].weight==1)
    p=parse_prompt('<lora:角色:.8>, <lora:画风:1.1>, (red hair:1.1), blue eyes:.8')
    check('two_independent_adapters',[(x.name,x.weight) for x in p.loras]==[('角色',.8),('画风',1.1)])
    check('controls_removed_from_text','lora:' not in p.text)
    check('explicit_word_weight',any(s.text=='red hair' and s.weight==1.1 for s in p.spans))
    check('bare_shorthand',any(s.text.strip()=='blue eyes' and s.weight==.8 for s in p.spans))
    check('nested_word_weights',abs(parse_prompt('(coat (buttons:1.1):.8)').spans[-1].weight-.88)<1e-7)
    check('same_duplicate_dedup',len(parse_prompt('<lora:a:.8><lora:a:0.8>').loras)==1)
    check('filename_with_spaces',parse_prompt('<lora:我的 角色.safetensors:.8>').loras[0].name=='我的 角色')
    check('ratios_preserved',parse_prompt('16:9').text=='16:9')
    for i,text in enumerate(['<lora:a:nan>','<lora:a:.8><lora:a:1.1>','<lora:../a:.8>','<lora:a:>','<lora:a:.8','(a:.8','a)','(a:nan)','(:.8)','(a:3)']):
        reject('invalid_tag_'+str(i),lambda text=text:parse_prompt(text))
    paths=runtime_paths(out/'sdxl_qnn',True)
    check('runtime_subfolder',paths['originals']==(out/'sdxl_qnn/Lora').resolve())
    check('cache_separate',paths['cache']==(out/'sdxl_qnn/.rin_lora/cache').resolve())
    tensors={'lora_unet_test.lora_down.weight':np.arange(8,dtype=np.float32).reshape(2,4),'lora_unet_test.lora_up.weight':np.arange(8,dtype=np.float32).reshape(4,2),'lora_unet_test.alpha':np.array(2,np.float32)}
    folder=paths['originals'];candidate=folder/'角色.safetensors'
    save_file(tensors,str(candidate),metadata={'ss_base_model_version':'sdxl_base_v1-0'})
    info=inspect_lora(candidate)
    check('safetensors_candidate_inspection',info['pairs'][0]['rank']==2)
    check('not_falsely_qnn_ready',info['qnn_ready'] is False and info['status']=='needs_qnn_template')
    save_file(tensors,str(folder/'sd15.safetensors'),metadata={'ss_base_model_version':'sd_v1'})
    reject('wrong_base_rejected',lambda:inspect_lora(folder/'sd15.safetensors'))
    (folder/'partial.safetensors.part').write_bytes(b'pending')
    (folder/'short.safetensors').write_bytes(b'new deliberately incomplete test fixture')
    check('partial_downloads_not_listed',len(scan_loras(out/'sdxl_qnn'))==3)
    reject('invalid_header_rejected',lambda:inspect_lora(folder/'short.safetensors'))
    save_file({'layer.lora_A.weight':np.ones((2,4),np.float32),'layer.lora_B.weight':np.ones((4,2),np.float32)},str(folder/'peft.safetensors'))
    check('peft_schema_inspection',inspect_lora(folder/'peft.safetensors')['pairs'][0]['format']=='peft')
    a,b=tensors['lora_unet_test.lora_down.weight'],tensors['lora_unet_test.lora_up.weight'];base=np.eye(4,dtype=np.float32);original=base.copy()
    check('training_alpha_distinct',np.allclose(lora_delta(a,b,4,.8),(b@a)*1.6))
    for weight in [0,.8,1.1,0]:
        merged=base+lora_delta(a,b,None,weight)
        check('merge_strength_'+str(weight)+'_'+str(len(results)),np.allclose(merged,base+weight*(b@a)))
    check('base_unchanged',np.array_equal(original,base))
    d2=lora_delta(a+1,b-2,2,1.1)
    check('two_independent_deltas',np.allclose(lora_delta(a,b,2,.8)+d2,.8*(b@a)+1.1*((b-2)@(a+1))))
    check('zero_first_preserves_second',np.array_equal(lora_delta(a,b,2,0)+d2,d2))
    reject('invalid_rank',lambda:lora_delta(a,np.ones((4,3),np.float32),2))
    reject('nonfinite_strength',lambda:lora_delta(a,b,2,math.nan))
    report={'scope':'host parser, Safetensors schema and numeric LoRA delta only; no APP/phone validation','total':len(results),'passed':sum(x['passed'] for x in results),'checks':results}
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SUMMARY',report['passed'], '/',report['total'])
    return 0 if report['passed']==report['total'] else 1
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',required=True);args=ap.parse_args();raise SystemExit(run(args.out_dir))
