from pathlib import Path
import os,sys,json,traceback,subprocess,yaml
import numpy as np,onnx
from onnx import helper,numpy_helper,TensorProto
from safetensors.numpy import save_file
r=Path(sys.argv[1]);sdk=r/'toolchain/sdxl-build/qairt-sdk/extracted-2.48.0.260626/qairt/2.48.0.260626';out=r/'toolchain/temp/v160-lora-adapters/linear01'
out.mkdir(parents=True,exist_ok=True);(out/'tmp').mkdir()
os.environ.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',QAIRT_TMP_DIR=str(out/'tmp'),TMP=str(out/'tmp'),TEMP=str(out/'tmp'))
os.environ['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+os.environ.get('PATH','');report={'scope':'synthetic linear adapter feasibility; no hardware inference'}
def step(s):(out/'status.txt').write_text(s);print(s,flush=True)
def call(label,cmd):
 step(label)
 with (out/(label+'.log')).open('w') as f:
  proc=subprocess.run(list(map(str,cmd)),cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120,creationflags=0x08000000|0x4000)
 report[label+'_rc']=proc.returncode
 if proc.returncode:raise RuntimeError(label+' rc='+str(proc.returncode))
try:
 rng=np.random.default_rng(160);base=rng.normal(0,.2,(4,4)).astype(np.float32);a=rng.normal(0,.1,(4,2)).astype(np.float32);b=rng.normal(0,.1,(2,4)).astype(np.float32)
 nodes=[helper.make_node('MatMul',['sample','base_weight'],['base_out']),helper.make_node('MatMul',['sample','lora_A'],['rank_out']),helper.make_node('Mul',['rank_out','lora_alpha'],['scaled']),helper.make_node('MatMul',['scaled','lora_B'],['delta']),helper.make_node('Add',['base_out','delta'],['output'])]
 inputs=[helper.make_tensor_value_info('sample',TensorProto.FLOAT,[1,4]),helper.make_tensor_value_info('lora_alpha',TensorProto.FLOAT,[1])]
 outputs=[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,4])]
 tensors=[numpy_helper.from_array(x,n) for n,x in [('base_weight',base),('lora_A',a),('lora_B',b)]]
 graph=helper.make_graph(nodes,'linear_lora',inputs,outputs,tensors)
 m=helper.make_model(graph,opset_imports=[helper.make_operatorsetid('',17)]);m.ir_version=9;onnx.checker.check_model(m);onnx.save(m,out/'linear.onnx')
 (out/'weights.txt').write_text('lora_A\nlora_B\n');wrapper="import qairt,runpy,sys;p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name='__main__')"
 call('CONVERT',[sys.executable,'-c',wrapper,sdk/'bin/x86_64-windows-msvc/qairt-converter','-i',out/'linear.onnx','--output_path',out/'linear.dlc','--lora_weight_list',out/'weights.txt','--quant_updatable_mode','float_only','--float_bitwidth','16','--onnx_skip_simplification'])
 cases=[]
 for name,fa,fb in [('original',1,1),('changed',1.5,-.7),('zero',0,0)]:
  file=out/(name+'.safetensors');save_file({'lora_A':a*fa,'lora_B':b*fb},str(file));cases.append({'name':name,'lora_weights':str(file)})
 cfg=out/'importer.yaml';cfg.write_text(yaml.safe_dump({'use_case':cases}))
 call('IMPORT',[sys.executable,'-c',wrapper,sdk/'bin/x86_64-windows-msvc/qairt-lora-importer','--input_network',out/'linear.onnx','--input_dlc',out/'linear.dlc','--lora_config',cfg,'--output_dir',out/'imported'])
 step('COMPILE_UPDATEABLE');import qairt
 config=yaml.safe_load((out/'imported/lora_output_files.yaml').read_text())
 for item in config['use_case']:item['enable_weights_updates']=True
 file=out/'adapters.yaml';file.write_text(yaml.safe_dump(config));model=qairt.load(str(out/'linear.dlc'));model.lora_use_cases=str(file)
 compiled=qairt.compile(model,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'));compiled.save(str(out/'compiled/linear.bin'))
 report['ok']=True;report['files']=[{'name':file.name,'bytes':file.stat().st_size} for file in (out/'compiled').glob('*.bin')];step('COMPLETE')
except Exception as e:report.update(ok=False,error=str(e),traceback=traceback.format_exc());traceback.print_exc();step('FAILED')
(out/'report.json').write_text(json.dumps(report,indent=2))
