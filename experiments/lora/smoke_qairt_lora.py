"""Tiny numerical/QAIRT LoRA proof; not WAI or Android hardware validation."""
from pathlib import Path
import argparse,json,os,subprocess,sys,traceback
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper
import onnxruntime as ort

ap=argparse.ArgumentParser();ap.add_argument('--sdk-root',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
sdk=Path(args.sdk_root);out=Path(args.out_dir)
if any((out/name).exists() for name in ['tiny.onnx','tiny.dlc','tiny.bin','result.json']):
 raise SystemExit('Prior experiment outputs exist; use a new output directory')
out.mkdir(parents=True,exist_ok=True)
status=out/'status.txt';report={"scope":"tiny CPU ONNX and offline SM8750 HTP compilation; no phone inference"}
def stage(name):status.write_text(name,encoding='utf-8');print(name,flush=True)
def save(): (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def cmd(label,argv):
 stage(label)
 with (out/(label+'.log')).open('w',encoding='utf-8') as log:
  p=subprocess.run([str(x) for x in argv],stdout=log,stderr=subprocess.STDOUT,cwd=out,env=os.environ.copy(),timeout=120,creationflags=0x08000000|0x4000)
 report[label+'_rc']=p.returncode;save()
 if p.returncode:raise RuntimeError(label+' failed; see '+label+'.log')
try:
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
 for key in ['QAIRT_SDK_ROOT','QNN_SDK_ROOT']:os.environ[key]=str(sdk)
 tmp=out/'tmp';tmp.mkdir(exist_ok=True)
 for key in ['QAIRT_TMP_DIR','TMP','TEMP']:os.environ[key]=str(tmp)
 os.environ['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+os.environ.get('PATH','')
 stage('ONNX_NUMERICS')
 rng=np.random.default_rng(160)
 base=rng.normal(0,.2,(4,4,1,1)).astype(np.float32)
 a=rng.normal(0,.1,(2,4,1,1)).astype(np.float32)
 b=rng.normal(0,.1,(4,2,1,1)).astype(np.float32)
 tensors=[numpy_helper.from_array(v,n) for n,v in [('base_weight',base),('lora_A',a),('lora_B',b)]]
 nodes=[helper.make_node('Conv',['sample','base_weight'],['base_out'],name='base_conv',kernel_shape=[1,1]),helper.make_node('Conv',['sample','lora_A'],['low_rank'],name='lora_down',kernel_shape=[1,1]),helper.make_node('Mul',['low_rank','lora_alpha'],['scaled'],name='lora_strength'),helper.make_node('Conv',['scaled','lora_B'],['delta'],name='lora_up',kernel_shape=[1,1]),helper.make_node('Add',['base_out','delta'],['output'],name='apply_lora')]
 graph=helper.make_graph(nodes,'rin_lora_tiny',[helper.make_tensor_value_info('sample',TensorProto.FLOAT,[1,4,2,2]),helper.make_tensor_value_info('lora_alpha',TensorProto.FLOAT,[1])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,4,2,2])],tensors)
 model=helper.make_model(graph,opset_imports=[helper.make_operatorsetid('',17)]);model.ir_version=9;onnx.checker.check_model(model);onnx.save(model,out/'tiny.onnx')
 opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
 session=ort.InferenceSession(str(out/'tiny.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
 x=rng.normal(size=(1,4,2,2)).astype(np.float32);d=b[:,:,0,0]@a[:,:,0,0];baseline=np.einsum('oi,nihw->nohw',base[:,:,0,0],x)
 errors=[]
 for strength in [0,.8,1.1,-.4,0]:
  y=session.run(None,{'sample':x,'lora_alpha':np.array([strength],np.float32)})[0]
  expected=np.einsum('oi,nihw->nohw',base[:,:,0,0]+strength*d,x)
  np.testing.assert_allclose(y,expected,rtol=2e-6,atol=2e-6)
  if strength==0:np.testing.assert_allclose(y,baseline,rtol=2e-6,atol=2e-6)
  errors.append({'strength':strength,'max_abs_error':float(np.max(np.abs(y-expected)))})
 report['onnx_numeric_checks']=errors;save()
 (out/'lora_weights.txt').write_text('lora_A\nlora_B\n',encoding='utf-8')
 wrapper="import qairt,runpy,sys; target=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(target,run_name='__main__')"
 cmd('CONVERT_UPDATEABLE',[sys.executable,'-c',wrapper,sdk/'bin/x86_64-windows-msvc/qairt-converter','-i',out/'tiny.onnx','--output_path',out/'tiny.dlc','--lora_weight_list',out/'lora_weights.txt','--quant_updatable_mode','float_only','--float_bitwidth','16','--onnx_skip_simplification'])
 stage('DLC_INSPECTION')
 import qairt
 from qti.aisw.converters.common import modeltools
 reader=modeltools.IrDlcReader();reader.open(str(out/'tiny.dlc'));names=list(reader.get_ir_graph_names());g=reader.get_ir_graph(names[0]);report['updateable_static_tensors']=[name for name,t in g.get_tensor_map().items() if t.is_updateable() and t.is_static_tensor()]
 if len(report['updateable_static_tensors'])<2:raise RuntimeError('LoRA weights not retained as updateable')
 save();stage('COMPILE_HTP')
 converted=qairt.load(str(out/'tiny.dlc'))
 compiled=qairt.compile(converted,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'))
 compiled.save(str(out/'tiny.bin'));report['context_bytes']=(out/'tiny.bin').stat().st_size;save()
 cmd('CONTEXT_INFO',[sdk/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe','--context_binary='+str(out/'tiny.bin'),'--json_file='+str(out/'tiny.context.json')])
 report['outcome']='offline_updateable_context_created';save();stage('COMPLETE_OFFLINE_ONLY')
except Exception as exc:
 report['error']=str(exc);report['traceback']=traceback.format_exc();report['outcome']='failed_at_'+status.read_text();save();stage('FAILED');traceback.print_exc();sys.exit(1)
