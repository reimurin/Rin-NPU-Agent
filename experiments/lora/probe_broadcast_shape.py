from pathlib import Path
import argparse,os,subprocess,json,traceback,yaml
ap=argparse.ArgumentParser();ap.add_argument('--root');args=ap.parse_args();r=Path(args.root);w=r/'development/v1.6.0';sdk=r/'toolchain/sdxl-build/qairt-sdk/extracted-2.48.0.260626/qairt/2.48.0.260626';out=r/'toolchain/temp/v160-lora-adapters/alpha4d01';out.mkdir(parents=True,exist_ok=True)
import sys
report={}
def state(s):(out/'status.txt').write_text(s);print(s,flush=True)
def run(label,cmd):
 state(label)
 with (out/(label+'.log')).open('w') as f:p=subprocess.run(list(map(str,cmd)),cwd=out,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=120,creationflags=0x08000000|0x4000)
 report[label]=p.returncode
 if p.returncode:raise RuntimeError(label+' failed')
try:
 env=os.environ.copy();env.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');env['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','')
 source=(w/'experiments/lora/smoke_qairt_lora.py').read_text()
 old="helper.make_tensor_value_info('lora_alpha',TensorProto.FLOAT,[1])";assert source.count(old)==1;source=source.replace(old,"helper.make_tensor_value_info('lora_alpha',TensorProto.FLOAT,[1,1,1,1])")
 old="np.array([strength],np.float32)";assert source.count(old)==1;source=source.replace(old,"np.array([strength],np.float32).reshape(1,1,1,1)")
 smoke=out/'smoke.py';smoke.write_text(source);base=out/'base'
 run('CONVERT',[sys.executable,'-B',smoke,'--sdk-root',sdk,'--out-dir',base])
 prev=r/'toolchain/temp/v160-lora-adapters/run01';cases=[{'name':n,'lora_weights':str(prev/(n+'.safetensors'))} for n in ['original','changed','zero']]
 cfg=out/'import.yaml';cfg.write_text(yaml.safe_dump({'use_case':cases}));wrapper="import qairt,runpy,sys;p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name='__main__')"
 run('IMPORT',[sys.executable,'-c',wrapper,sdk/'bin/x86_64-windows-msvc/qairt-lora-importer','--input_network',base/'tiny.onnx','--input_dlc',base/'tiny.dlc','--lora_config',cfg,'--output_dir',out/'imported'])
 state('GENERATE');os.environ.update(env);os.environ['QAIRT_TMP_DIR']=str(out/'tmp');(out/'tmp').mkdir()
 import qairt
 cfgdata=yaml.safe_load((out/'imported/lora_output_files.yaml').read_text())
 for c in cfgdata['use_case']:c['enable_weights_updates']=True
 config=out/'adapters.yaml';config.write_text(yaml.safe_dump(cfgdata));model=qairt.load(str(base/'tiny.dlc'));model.lora_use_cases=str(config)
 compiled=qairt.compile(model,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'));compiled.save(str(out/'compiled/tiny.bin'))
 report['ok']=True;report['files']=[{'name':p.name,'bytes':p.stat().st_size} for p in (out/'compiled').glob('*.bin')];state('COMPLETE')
except Exception as e:report.update(ok=False,error=str(e),traceback=traceback.format_exc());state('FAILED');traceback.print_exc()
(out/'report.json').write_text(json.dumps(report,indent=2))
