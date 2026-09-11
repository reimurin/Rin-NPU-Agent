from pathlib import Path
import os,sys,json,hashlib,traceback,subprocess
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
r=Path(sys.argv[1]);sdk=r/'toolchain/sdxl-build/qairt-sdk/extracted-2.48.0.260626/qairt/2.48.0.260626';out=r/'toolchain/temp/v160-lora-dynamic/run01';out.mkdir(parents=True,exist_ok=True);(out/'tmp').mkdir()
os.environ.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',QAIRT_TMP_DIR=str(out/'tmp'),TEMP=str(out/'tmp'),TMP=str(out/'tmp'));os.environ['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+os.environ.get('PATH','')
report={'scope':'dynamic A/B input LoRA; CPU and offline V79 compile, not phone validation'}
def state(s):(out/'status.txt').write_text(s);print(s,flush=True)
try:
 state('BUILD_GRAPH');rng=np.random.default_rng(160);base=rng.normal(0,.08,(32,32)).astype(np.float32);a=rng.normal(0,.15,(32,8)).astype(np.float32);b=rng.normal(0,.15,(8,32)).astype(np.float32);x=np.linspace(-1,1,256,dtype=np.float32).reshape(1,8,32)
 shapes={'sample':[1,8,32],'lora_A':[32,8],'lora_B':[8,32],'lora_alpha':[1,1,1]}
 nodes=[helper.make_node('MatMul',['sample','base_weight'],['base_out']),helper.make_node('MatMul',['sample','lora_A'],['low_rank']),helper.make_node('MatMul',['low_rank','lora_B'],['delta']),helper.make_node('Mul',['delta','lora_alpha'],['weighted_delta']),helper.make_node('Add',['base_out','weighted_delta'],['output'])]
 inputs=[helper.make_tensor_value_info(n,TensorProto.FLOAT,s) for n,s in shapes.items()];outputs=[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,8,32])];graph=helper.make_graph(nodes,'dynamic_lora',inputs,outputs,[numpy_helper.from_array(base,'base_weight')]);model=helper.make_model(graph,opset_imports=[helper.make_operatorsetid('',17)]);model.ir_version=9;onnx.checker.check_model(model);onnx.save(model,out/'dynamic.onnx')
 opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1;session=ort.InferenceSession(str(out/'dynamic.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
 assets=out/'assets';assets.mkdir();x.tofile(assets/'sample.raw');manifest={'schema':1,'soc':69,'dsp':79,'context':'dynamic.bin','route':'dynamic_weights','scope':report['scope'],'cases':[],'assets':[]}
 for name,alpha,kind in [('base0',0,'original'),('original08',.8,'original'),('original11',1.1,'original'),('changed08',.8,'changed'),('restored08',.8,'original'),('restored0',0,'original'),('zero11',1.1,'zero')]:
  aa=a*(1.5 if kind=='changed' else 0 if kind=='zero' else 1);bb=b*(-.7 if kind=='changed' else 0 if kind=='zero' else 1);scale=np.array([alpha],np.float32).reshape(1,1,1)
  value=session.run(None,{'sample':x,'lora_A':aa,'lora_B':bb,'lora_alpha':scale})[0];expected=x@base+alpha*((x@aa)@bb);np.testing.assert_allclose(value,expected,rtol=2e-5,atol=2e-6)
  value.tofile(assets/(name+'.expected.raw'));aa.tofile(assets/(name+'.A.raw'));bb.tofile(assets/(name+'.B.raw'));scale.tofile(assets/(name+'.alpha.raw'))
  manifest['cases'].append({'name':name,'alpha':alpha,'weights':kind,'elements':256,'atol':0.003,'rtol':0.025,'inputs':{'sample':'sample.raw','lora_A':name+'.A.raw','lora_B':name+'.B.raw','lora_alpha':name+'.alpha.raw'}})
 state('CONVERT');wrapper="import qairt,runpy,sys;p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name='__main__')"
 with (out/'convert.log').open('w') as log:
  p=subprocess.run([sys.executable,'-c',wrapper,str(sdk/'bin/x86_64-windows-msvc/qairt-converter'),'-i',str(out/'dynamic.onnx'),'--output_path',str(out/'dynamic.dlc'),'--float_bitwidth','16','--onnx_skip_simplification'],stdout=log,stderr=subprocess.STDOUT,timeout=120,creationflags=0x08000000|0x4000)
 assert p.returncode==0,'DLC conversion failed';state('COMPILE_V79');import qairt
 converted=qairt.load(str(out/'dynamic.dlc'));compiled=qairt.compile(converted,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'));compiled.save(str(assets/'dynamic.bin'))
 p=subprocess.run([str(sdk/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe'),'--context_binary='+str(assets/'dynamic.bin'),'--json_file='+str(out/'context.json')],capture_output=True,text=True,timeout=20,creationflags=0x08000000);assert p.returncode==0,p.stderr
 meta=json.loads((out/'context.json').read_text());assert meta['info']['socModel']==69 and meta['info']['contextMetadata']['info']['dspArch']==79
 graph=meta['info']['graphs'][0]['info'];actual={t['info']['name']:t['info']['dimensions'] for t in graph['graphInputs']};assert actual==shapes,(actual,shapes)
 for f in sorted(assets.iterdir()):manifest['assets'].append({'name':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 (assets/'manifest.json').write_text(json.dumps(manifest,indent=2));report.update(ok=True,cpu_cases=7,inputs=actual,context_bytes=(assets/'dynamic.bin').stat().st_size);state('COMPLETE')
except Exception as exc:report.update(ok=False,error=str(exc),traceback=traceback.format_exc());traceback.print_exc();state('FAILED')
(out/'report.json').write_text(json.dumps(report,indent=2))
