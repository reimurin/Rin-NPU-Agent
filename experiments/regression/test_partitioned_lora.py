"""Actual partition loader/executor tested against ONNX Runtime CPU references."""
from pathlib import Path
import argparse,ast,copy,json,sys,tempfile,unittest
import numpy as np
import onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from safetensors.numpy import save_file
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/main/assets/sdxl_runtime'),str(ROOT/'model_tools/partitioned')]
import rin_lora as common
import rin_lora_partitioned as runtime
from independent_lora import inject
from rin_tensor_io import write_input_rows,read_float_output,MANIFEST_NAME

class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='case_',dir=OUT);self.base=Path(self.temp.name);self.folder=self.base/'Lora';self.folder.mkdir();self.work=self.base/'work';self.work.mkdir()
  self.template=self.base/runtime.RELATIVE;self.template.mkdir(parents=True)
  self.manifest={'schema':1,'format':runtime.FORMAT,'model_id':runtime.MODEL_ID,'base_model':'WAI-illustrious-SDXL-v170','complete':True,'soc':69,'dsp':79,'resolution':[1024,1024],'rank_capacity':64,'graphs':{}}
  self.models={};self.weights={};self.layers={};self.calls=[];self.readers={}
  for stage in ['encoder','decoder']:
   parts=[];count=2 if stage=='encoder' else 1
   for index in range(count):
    pid=stage+'_p'+str(index);inf=4 if index==0 else 3;outf=3 if index==0 else 2;shape=[1,2,inf];outshape=[1,2,outf]
    input_name='sample' if index==0 else 'bridge';output_name='bridge' if stage=='encoder' and index==0 else 'output'
    weight=(np.arange(inf*outf,dtype=np.float32).reshape(inf,outf)-3)/30;self.weights[pid]=weight
    nodes=[helper.make_node('MatMul',[input_name,'weight'],['linear'],name='/'+stage+'/proj'+str(index)+'/MatMul'),helper.make_node('Relu',['linear'],['positive']),helper.make_node('Add',['positive','bias'],[output_name],name='plus_bias')]
    outputs=[helper.make_tensor_value_info(output_name,TensorProto.FLOAT,outshape)]
    if stage=='encoder' and index==0:
     nodes.append(helper.make_node('Identity',['positive'],['skip']));outputs.append(helper.make_tensor_value_info('skip',TensorProto.FLOAT,outshape))
    graph=helper.make_graph(nodes,pid,[helper.make_tensor_value_info(input_name,TensorProto.FLOAT,shape)],outputs,[numpy_helper.from_array(weight,'weight'),numpy_helper.from_array(np.array(.125,np.float32),'bias')])
    model=helper.make_model(graph,opset_imports=[helper.make_operatorsetid('',17)]);model.ir_version=9;model,spec=inject(model,64)
    layer=spec['layers'][0];layer['aliases'].append('sgm_'+pid);self.layers[pid]=layer
    actual_output=output_name+'_permute' if index==0 else output_name
    for node in model.graph.node:
     for field in [node.input,node.output]:
      for i,name in enumerate(field):
       if name==output_name:field[i]=actual_output
    model.graph.output[0].name=actual_output;onnx.checker.check_model(model)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    self.models[pid]=ort.InferenceSession(model.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
    file=self.template/(pid+'.bin');file.write_bytes(('fixture-'+pid).encode())
    inputs=[{'name':input_name,'shape':shape}];logical_outputs=[{'name':output_name,'shape':outshape}]
    if len(outputs)>1:logical_outputs.append({'name':'skip','shape':outshape})
    def binding(name,actual,shape):return {'logical':name,'name':actual,'shape':shape,'qnn_shape':shape,'qnn_dtype':'QNN_DATATYPE_FLOAT_16','to_qnn_axes':list(range(len(shape)))}
    ib=[binding(input_name,input_name,shape)];ob=[binding(output_name,actual_output,outshape)]
    if len(outputs)>1:ob.append(binding('skip','skip',outshape))
    qi={input_name:shape,layer['a']:layer['a_shape'],layer['b']:layer['b_shape']}
    part={'id':pid,'context_file':file.name,'context_bytes':file.stat().st_size,'context_sha256':runtime.digest(file),'layers':[layer],'inputs':inputs,'outputs':logical_outputs,'input_bindings':ib,'output_bindings':ob,'qnn_inputs':qi,'qnn_outputs':{b['name']:b['qnn_shape'] for b in ob}}
    parts.append(part)
   self.manifest['graphs'][stage]={'parts':parts,'layers':[p['layers'][0] for p in parts],'original_inputs':['sample'],'original_outputs':['output','skip'] if stage=='encoder' else ['output']}
  self.mp=self.template/'lora_template.json';self.save_manifest()
 def tearDown(self):self.temp.cleanup()
 def save_manifest(self):self.mp.write_text(json.dumps(self.manifest))
 def adapter(self,name='a',targets=None,rank=2,alpha=3.,extra=None):
  values={};record={}
  for pid in targets or ['encoder_p0','encoder_p1']:
   layer=self.layers[pid];i,o=layer['in_features'],layer['out_features'];A=(np.arange(rank*i,dtype=np.float32).reshape(rank,i)-2)/20;B=(np.arange(o*rank,dtype=np.float32).reshape(o,rank)+1)/25
   prefix='sgm_'+pid;values[prefix+'.lora_down.weight']=A;values[prefix+'.lora_up.weight']=B
   if alpha is not None:values[prefix+'.alpha']=np.array(alpha,np.float32)
   record[pid]=(A,B,rank if alpha is None else alpha)
  if extra:values.update(extra)
  file=self.folder/(name+'.safetensors');save_file(values,str(file),metadata={'ss_base_model_version':'sdxl_base_v1-0'});self.readers[name]=record;return file
 def plan(self,prompt='<lora:a:.8>'):return common.prepare(prompt,'',self.base,1024,1024,log=lambda _:None)
 def single(self,context,listing,output,**kw):
  pid=Path(context).stem;self.calls.append((pid,kw));model=self.models[pid];dest=Path(output);dest.mkdir(parents=True)
  for rid,line in enumerate(Path(listing).read_text().splitlines()):
   entries=dict(token.split(':=',1) for token in line.split());feed={inp.name:np.fromfile(entries[inp.name],'<f4').reshape(inp.shape) for inp in model.get_inputs()}
   arrays=model.run(None,feed);result=dest/('Result_'+str(rid));result.mkdir();records=[]
   for info,array in zip(model.get_outputs(),arrays):
    file=result/(info.name+'.raw');array.astype('<f4').tofile(file);records.append({'name':info.name,'file':file.name,'dtype':'float32','shape':list(array.shape),'bytes':file.stat().st_size})
   (result/MANIFEST_NAME).write_text(json.dumps({'schema':1,'complete':True,'outputs':records}))
  return 3.0
 def run_plan(self,plan,rows=2,**kwargs):
  x=np.linspace(-1,1,8,dtype=np.float32).reshape(1,2,4);files=[]
  for index in range(rows):
   file=self.work/('x'+str(index)+'.raw');(x*(index+1)).astype('<f2' if kwargs.get('native_input') else '<f4').tofile(file);files.append(['sample:='+str(file)])
  listing=self.work/'inputs.txt';write_input_rows(listing,files);out=self.work/'output'
  elapsed=plan.execute('encoder',listing,out,self.single,work_dir=self.work,**kwargs);return out,elapsed,x
 def expected(self,x,selections):
  state=x;skip=None
  for pid in ['encoder_p0','encoder_p1']:
   weight=self.weights[pid].copy()
   for name,strength in selections:
    if pid in self.readers[name]:
     A,B,alpha=self.readers[name][pid];weight+=strength*alpha/A.shape[0]*(A.T@B.T)
   state=np.maximum(state@weight,0)
   if skip is None:skip=state.copy()
   state=state+.125
  return state,skip
 def test_01_actual_loader_to_partitioned_onnx(self):
  self.adapter();out,elapsed,x=self.run_plan(self.plan());self.assertEqual(elapsed,6)
  for row in range(2):
   expected,skip=self.expected(x*(row+1),[('a',.8)])
   np.testing.assert_allclose(read_float_output(out/('Result_'+str(row)),'output',[1,2,2]),expected,rtol=1e-5,atol=1e-6)
   np.testing.assert_allclose(read_float_output(out/('Result_'+str(row)),'skip',[1,2,3]),skip,rtol=1e-5,atol=1e-6)
  self.assertEqual(len(self.calls),2)
 def test_02_change_strength(self):
  self.adapter();out,_,x=self.run_plan(self.plan('<lora:a:1.1>'));expected,_=self.expected(x,[('a',1.1)]);np.testing.assert_allclose(read_float_output(out/'Result_0','output',[1,2,2]),expected,rtol=1e-5,atol=1e-6)
 def test_03_two_adapters_no_cross_terms(self):
  self.adapter('a',rank=2);self.adapter('b',rank=3);out,_,x=self.run_plan(self.plan('<lora:a:.8><lora:b:-.4>'));expected,_=self.expected(x,[('a',.8),('b',-.4)]);np.testing.assert_allclose(read_float_output(out/'Result_0','output',[1,2,2]),expected,rtol=1e-5,atol=1e-6)
 def test_04_unused_stage_keeps_base(self):
  self.adapter();plan=self.plan();self.assertNotIn('decoder',plan.contexts);self.assertEqual(plan.inputs,{})
 def test_05_no_tags_no_component_required(self):
  plan=common.prepare('cat','blur',self.base/'missing',1024,1024);self.assertEqual(plan.contexts,{});self.assertEqual(plan.prompt,'cat')
 def test_06_zero_strength_removes_control(self):
  plan=common.prepare('cat <lora:absent:0>','',self.base/'missing',1024,1024);self.assertEqual(plan.contexts,{});self.assertEqual(plan.prompt,'cat ')
 def test_07_cache_reuse(self):
  self.adapter();a=self.plan();b=self.plan();self.assertEqual(a.banks,b.banks)
 def test_08_corrupt_cache_rebuilt(self):
  self.adapter();a=self.plan();f=Path(next(iter(a.banks['encoder_p0'].values())));f.write_bytes(b'x'*f.stat().st_size);b=self.plan();self.assertNotEqual(a.banks,b.banks)
 def test_09_context_corruption_rejected(self):
  self.adapter();file=self.template/'encoder_p0.bin';file.write_bytes(b'x'*file.stat().st_size)
  with self.assertRaises(common.LoraError):self.plan()
 def test_10_unknown_alias_rejected(self):
  self.adapter(extra={'lora_te1_q.lora_down.weight':np.ones((2,4),np.float32),'lora_te1_q.lora_up.weight':np.ones((3,2),np.float32)})
  with self.assertRaises(common.LoraError):self.plan()
 def test_11_rank_capacity_enforced(self):
  self.adapter('a',rank=40);self.adapter('b',rank=40)
  with self.assertRaises(common.LoraError):self.plan('<lora:a:1><lora:b:1>')
 def test_12_negative_control_rejected(self):
  with self.assertRaises(common.LoraError):common.prepare('cat','<lora:a:1>',self.base,1024,1024)
 def test_13_missing_binding_rejected(self):
  self.adapter();self.manifest['graphs']['encoder']['parts'][0]['output_bindings'].pop();self.save_manifest()
  with self.assertRaises(common.LoraError):self.plan()
 def test_14_same_size_wrong_shape_rejected(self):
  self.adapter();self.manifest['graphs']['encoder']['parts'][0]['output_bindings'][0]['qnn_shape']=[1,3,2];self.save_manifest()
  with self.assertRaises(common.LoraError):self.plan()
 def test_15_duplicate_alias_rejected(self):
  self.adapter();parts=self.manifest['graphs']['encoder']['parts'];parts[1]['layers'][0]['aliases'].append('sgm_encoder_p0');self.save_manifest()
  with self.assertRaises(common.LoraError):self.plan()
 def test_16_incomplete_component_rejected(self):
  self.adapter();self.manifest['complete']=False;self.save_manifest()
  with self.assertRaises(common.LoraError):self.plan()
 def test_17_native_input_promoted_correctly(self):
  self.adapter();out,_,x=self.run_plan(self.plan(),native_input=True);x=x.astype(np.float16).astype(np.float32);expected,_=self.expected(x,[('a',.8)]);np.testing.assert_allclose(read_float_output(out/'Result_0','output',[1,2,2]),expected,rtol=1e-5,atol=1e-6)
 def test_18_native_output_contract(self):
  self.adapter();out,_,_=self.run_plan(self.plan(),native=True);self.assertTrue((out/'Result_0/output_native.raw').is_file());self.assertTrue((out/'Result_0/output.raw').is_file())
 def test_19_fresh_outputs_replace_only_owned_work(self):
  self.adapter();plan=self.plan();out,_,_=self.run_plan(plan);(self.work/'keep.txt').write_text('preserve');self.run_plan(plan);self.assertEqual((self.work/'keep.txt').read_text(),'preserve')
 def test_20_output_outside_work_rejected(self):
  self.adapter();plan=self.plan();file=self.work/'x.raw';np.ones((1,2,4),np.float32).tofile(file);listing=self.work/'inputs.txt';write_input_rows(listing,[['sample:='+str(file)]])
  with self.assertRaises(ValueError):plan.execute('encoder',listing,self.base/'outside',self.single,work_dir=self.work)
 def test_21_failed_part_produces_no_final_manifest(self):
  self.adapter();plan=self.plan();real=self.single
  def failed(context,*args,**kwargs):
   if Path(context).stem=='encoder_p1':raise RuntimeError('simulated NPU failure')
   return real(context,*args,**kwargs)
  self.single=failed
  with self.assertRaises(RuntimeError):self.run_plan(plan)
  self.assertFalse((self.work/'output/Result_0'/MANIFEST_NAME).exists())
 def test_22_plain_qnn_wrapper_does_not_use_foreign_interface(self):
  source=(ROOT/'src/main/assets/sdxl_runtime/phone_generate.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qnn_run');calls=[]
  namespace={'_ACTIVE_LORA_SESSION':common.Plan('','',{}, {},{}),'_qnn_run_single':lambda *a,**k:calls.append((a,k)) or 7}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_wrapper','exec'),namespace)
  self.assertEqual(namespace['qnn_run']('base.bin','inputs','out'),7);self.assertEqual(len(calls),1)
 def test_23_all_original_named_outputs_retained(self):
  self.adapter();out,_,_=self.run_plan(self.plan());records=json.loads((out/'Result_0'/MANIFEST_NAME).read_text())['outputs'];self.assertEqual({x['name'] for x in records},{'output','skip'})
 def test_24_nonfinite_adapter_rejected(self):
  file=self.adapter();weights=common.SafeWeights(file);values={k:weights.read(k) for k in weights.tensors};values['sgm_encoder_p0.lora_down.weight'][0,0]=np.nan;save_file(values,str(file))
  with self.assertRaises(common.LoraError):self.plan()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();OUT=Path(args.out)
 if OUT.exists():raise SystemExit('Choose new test directory')
 OUT.mkdir(parents=True)
 with (OUT/'tests.log').open('w',encoding='utf-8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 report={'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'scope':'actual loader/executor and ONNX CPU numerical reference, synthetic model contexts, not handset NPU'}
 (OUT/'result.json').write_text(json.dumps(report,indent=2));print((OUT/'tests.log').read_text());print(report);raise SystemExit(not result.wasSuccessful())
