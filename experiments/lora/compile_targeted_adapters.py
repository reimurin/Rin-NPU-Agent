from pathlib import Path
import argparse,os,json,traceback,shutil
import yaml
ap=argparse.ArgumentParser();ap.add_argument('--sdk');ap.add_argument('--base');ap.add_argument('--imported');ap.add_argument('--out');a=ap.parse_args();sdk,base,imp,out=map(Path,[a.sdk,a.base,a.imported,a.out]);out.mkdir(parents=True,exist_ok=True)
os.environ.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
tmp=out/'tmp';tmp.mkdir();os.environ.update(TEMP=str(tmp),TMP=str(tmp),QAIRT_TMP_DIR=str(tmp))
import qairt
summary=[]
for label,options in [('default',{}),('updatable',{'enable_weights_updates':True})]:
 d=out/label;d.mkdir();(out/'status.txt').write_text(label)
 cfg=yaml.safe_load((imp/'lora_output_files.yaml').read_text())
 for c in cfg['use_case']:c.update(options)
 cf=d/'adapter_weights.yaml';cf.write_text(yaml.safe_dump(cfg))
 item={'label':label,'options':options}
 try:
  model=qairt.load(str(base/'tiny.dlc'));model.lora_use_cases=str(cf)
  compiled=qairt.compile(model,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'))
  compiled.save(str(d/'tiny.bin'))
  item['module_path']=str(getattr(compiled.module,'path',''));item['assets']={str(k):str(getattr(v,'path',v)) for k,v in compiled.assets.items()}
  for asset in compiled.assets.values():
   src=Path(str(getattr(asset,'path','')))
   if src.is_file() and src.suffix=='.bin':
    dest=d/src.name
    if not dest.exists():shutil.copy2(src,dest)
  item['ok']=True;item['files']=[{'name':p.name,'bytes':p.stat().st_size} for p in d.glob('*.bin')]
 except Exception as e:item.update(ok=False,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
 summary.append(item);(out/'report.json').write_text(json.dumps(summary,indent=2));print('RESULT',label,item,flush=True)
(out/'status.txt').write_text('COMPLETE')
