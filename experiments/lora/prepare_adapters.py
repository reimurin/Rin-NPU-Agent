"""Generate tiny LoRA adapter sections with the pinned SDK; never touch WAI."""
from pathlib import Path
import argparse,json,os,subprocess,sys,traceback,hashlib
import numpy as np
from safetensors.numpy import save_file,load_file
import onnx
from onnx import numpy_helper
import yaml
ap=argparse.ArgumentParser();ap.add_argument('--sdk',required=True);ap.add_argument('--base',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
sdk,base,out=map(Path,[a.sdk,a.base,a.out]);out.mkdir(parents=True,exist_ok=True)
if (out/'report.json').exists():raise RuntimeError('Refusing previous result directory')
report={'scope':'tiny SM8750 LoRA section creation; no phone execution'}
def status(t): (out/'status.txt').write_text(t,encoding='utf-8');print(t,flush=True)
def save(): (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def run(label,cmd):
 status(label)
 with (out/(label+'.log')).open('w',encoding='utf-8') as f:
  p=subprocess.run(list(map(str,cmd)),cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120,creationflags=0x08000000|0x4000)
 report[label+'_rc']=p.returncode;save()
 if p.returncode:raise RuntimeError(label+' rc='+str(p.returncode))
try:
 env=os.environ.copy();env.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 env['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','')
 tmp=out/'tmp';tmp.mkdir();env.update(TMP=str(tmp),TEMP=str(tmp),QAIRT_TMP_DIR=str(tmp))
 status('PREPARE')
 model=onnx.load(str(base/'tiny.onnx'));weights={t.name:numpy_helper.to_array(t).copy() for t in model.graph.initializer}
 orig={n:weights[n] for n in ['lora_A','lora_B']};new={n:arr*factor for (n,arr),factor in zip(orig.items(),[1.5,-.7])};zero={n:np.zeros_like(arr) for n,arr in orig.items()}
 cases=[]
 for name,values in [('original',orig),('changed',new),('zero',zero)]:
  p=out/(name+'.safetensors');save_file(values,str(p));cases.append({'name':name,'lora_weights':str(p),'output_path':str(out/'imported')})
 cfg=out/'importer.yaml';cfg.write_text(yaml.safe_dump({'use_case':cases}),encoding='utf-8')
 wrapper="import qairt,runpy,sys; path=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(path,run_name='__main__')"
 run('IMPORT',[sys.executable,'-c',wrapper,sdk/'bin/x86_64-windows-msvc/qairt-lora-importer','--input_network',base/'tiny.onnx','--input_dlc',base/'tiny.dlc','--lora_config',cfg,'--output_dir',out/'imported'])
 cbg=yaml.safe_load((out/'imported/lora_output_files.yaml').read_text())
 for case in cbg['use_case']:
  case.pop('encodings',None);case['weights_only']=True;case['enable_weights_updates']=True
  vals=load_file(case['weights']);print('TRANSFORMED',case['name'],{k:list(v.shape) for k,v in vals.items()},flush=True)
 cbg_cfg=out/'adapter_weight_config.yaml';cbg_cfg.write_text(yaml.safe_dump(cbg),encoding='utf-8')
 run('GENERATE',[sdk/'bin/x86_64-windows-msvc/qnn-context-binary-generator.exe','--backend',sdk/'lib/x86_64-windows-msvc/QnnHtp.dll','--model',sdk/'lib/x86_64-windows-msvc/QnnModelDlc.dll','--dlc_path',base/'tiny.dlc','--binary_file','tiny_lora','--output_dir',out/'compiled','--htp_socs','sm8750','--adapter_weight_config',cbg_cfg,'--log_level','info'])
 report['files']=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (out/'compiled').rglob('*') if p.is_file()]
 if not any('original' in f['path'] and f['path'].endswith('.bin') for f in report['files']):raise RuntimeError('Adapter output missing')
 save();status('COMPLETE_SECTIONS_CREATED')
except Exception as e:
 report['error']=str(e);report['traceback']=traceback.format_exc();save();status('FAILED');traceback.print_exc();sys.exit(1)
