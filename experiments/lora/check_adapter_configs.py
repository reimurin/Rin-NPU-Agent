from pathlib import Path
import argparse,json,subprocess,os,hashlib
import yaml
ap=argparse.ArgumentParser();ap.add_argument('--sdk');ap.add_argument('--base');ap.add_argument('--imported');ap.add_argument('--out');a=ap.parse_args();sdk,base,src,out=map(Path,[a.sdk,a.base,a.imported,a.out]);out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','');env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
cases=yaml.safe_load((src/'lora_output_files.yaml').read_text())['use_case'];summary=[]
for label,opts in [('float_default',{}),('metadata_only',{'enable_weights_updates':True}),('weights_only',{'weights_only':True})]:
 d=out/label;d.mkdir(exist_ok=False);(out/'status.txt').write_text(label)
 config={'use_case':[dict(c,**opts) for c in cases]};cfg=d/'adapters.yaml';cfg.write_text(yaml.safe_dump(config),encoding='utf-8')
 args=[str(sdk/'bin/x86_64-windows-msvc/qnn-context-binary-generator.exe'),'--backend',str(sdk/'lib/x86_64-windows-msvc/QnnHtp.dll'),'--model',str(sdk/'lib/x86_64-windows-msvc/QnnModelDlc.dll'),'--dlc_path',str(base/'tiny.dlc'),'--binary_file','tiny_lora','--output_dir',str(d/'compiled'),'--htp_socs','sm8750','--adapter_weight_config',str(cfg),'--log_level','error']
 with (d/'generate.log').open('w') as log:
  p=subprocess.run(args,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60,creationflags=0x08000000|0x4000)
 item={'configuration':label,'rc':p.returncode,'options':opts,'files':[{'path':str(p.relative_to(d)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in d.rglob('*.bin')]};summary.append(item);(out/'results.json').write_text(json.dumps(summary,indent=2));print(label,p.returncode,flush=True)
(out/'status.txt').write_text('COMPLETE')
