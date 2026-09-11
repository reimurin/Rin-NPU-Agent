"""Actual per-layer loader and graph numerical tests; no QNN/image inference."""
from pathlib import Path
import argparse,ast,copy,json,sys,tempfile,unittest
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from safetensors.numpy import save_file
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/main/assets/sdxl_runtime'),str(ROOT/'scripts')]
import rin_lora as runtime
from build_wai_lora_layer_inputs import inject

class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='lora_case_',dir=OUT);self.root=Path(self.temp.name);(self.root/'Lora').mkdir()
  self.template=self.root/'lora_runtime/wai-linear-r32-1024';self.template.mkdir(parents=True)
  self.manifest={'schema':1,'format':'rin-lora-module-banks-v1','soc':69,'dsp':79,'base_model':'WAI-illustrious-SDXL-v170','resolution':[1024,1024],'rank_capacity':8,'compiled':True,'template_id':'synthetic-test','stages':{}}
  self.models={};self.weights={}
  for stage,inf,outf in [('encoder',4,3),('decoder',3,2)]:
   weight=np.arange(inf*outf,dtype=np.float32).reshape(inf,outf)/30;self.weights[stage]=weight
   n=helper.make_node('MatMul',['sample','W'],['output'],name='/'+stage+'/proj/MatMul')
   g=helper.make_graph([n],stage,[helper.make_tensor_value_info('sample',TensorProto.FLOAT,[1,2,inf])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,2,outf])],[numpy_helper.from_array(weight,'W')])
   m=helper.make_model(g,opset_imports=[helper.make_operatorsetid('',17)]);m.ir_version=9
   m,spec=inject(m,8);onnx.checker.check_model(m);self.models[stage]=m
   f=self.template/(stage+'.bin');f.write_bytes(stage.encode());spec.update(context=f.name,context_bytes=f.stat().st_size,context_sha256=runtime.digest(f))
   self.manifest['stages'][stage]=spec
  self.mp=self.template/'manifest.json';self.write_manifest()
 def tearDown(self):self.temp.cleanup()
 def write_manifest(self):self.mp.write_text(json.dumps(self.manifest))
 def adapter(self,name='a',rank=2,stage='encoder',dtype=np.float32,extra=None):
  inf,outf=self.weights[stage].shape;A=(np.arange(rank*inf,dtype=np.float32).reshape(rank,inf)+1)/20;B=(np.arange(outf*rank,dtype=np.float32).reshape(outf,rank)-1)/25
  prefix='lora_unet_'+stage+'_proj'
  t={prefix+'.lora_down.weight':A.astype(dtype),prefix+'.lora_up.weight':B.astype(dtype),prefix+'.alpha':np.array(rank*2,dtype=dtype)}
  if extra:t.update(extra)
  f=self.root/'Lora'/(name+'.safetensors');save_file(t,str(f),metadata={'ss_base_model_version':'sdxl_base_v1-0'});return A.astype(np.float32),B.astype(np.float32),f
 def plan(self,text):return runtime.prepare(text,'',self.root,1024,1024,log=lambda _:None)
 def bank(self,plan,stage='encoder'):
  self.assertEqual(len(plan.inputs[stage]),1);path=plan.inputs[stage][0].split(':=',1)[1];return np.fromfile(path,np.float32)
 def test_01_real_file_to_graph(self):
  A,B,_=self.adapter();plan=self.plan('cat <lora:a:.8>');bank=self.bank(plan)
  options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
  session=ort.InferenceSession(self.models['encoder'].SerializeToString(),sess_options=options,providers=['CPUExecutionProvider']);x=np.ones((1,2,4),np.float32)
  actual=session.run(None,{'sample':x,'rin_lora_m0000':bank})[0];expected=x@self.weights['encoder']+1.6*((x@A.T)@B.T)
  np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6);self.assertEqual(plan.prompt,'cat ');self.assertNotIn('decoder',plan.contexts)
 def test_02_two_adapters_no_cross_terms(self):
  A,B,_=self.adapter('a',2);C,D,_=self.adapter('b',3);bank=self.bank(self.plan('<lora:a:.8><lora:b:-.4>'))
  np.testing.assert_allclose(bank[:32].reshape(4,8)@bank[32:].reshape(8,3),1.6*(A.T@B.T)-.8*(C.T@D.T),rtol=1e-5,atol=1e-6)
 def test_03_zero_uses_original_context(self):
  p=runtime.prepare('cat <lora:any:0>','',OUT/'not-installed',1024,1024);self.assertFalse(p.contexts);self.assertFalse(p.inputs)
 def test_04_both_stages(self):
  self.adapter('a',2);self.adapter('b',2,'decoder');p=self.plan('<lora:a:.8><lora:b:1.1>');self.assertEqual(set(p.contexts),{'encoder','decoder'})
 def test_05_same_recipe_cache_reused(self):
  self.adapter();a=self.plan('<lora:a:1>');b=self.plan('<lora:a:1>');self.assertEqual(a.inputs,b.inputs)
 def test_06_strength_changes_cache(self):
  self.adapter();self.assertNotEqual(self.plan('<lora:a:.8>').inputs,self.plan('<lora:a:1.1>').inputs)
 def test_07_cache_corruption_regenerated(self):
  self.adapter();a=self.plan('<lora:a:1>');p=Path(a.inputs['encoder'][0].split(':=',1)[1]);p.write_bytes(b'x'*p.stat().st_size);b=self.plan('<lora:a:1>');self.assertNotEqual(a.inputs,b.inputs);self.assertTrue(np.isfinite(self.bank(b)).all())
 def test_08_context_corruption_blocked(self):
  self.adapter();f=self.template/'encoder.bin';f.write_bytes(b'x'*f.stat().st_size)
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1>')
 def test_09_capacity_not_truncated(self):
  self.adapter('a',5);self.adapter('b',5)
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1><lora:b:1>')
 def test_10_unsupported_whole_adapter_rejected(self):
  self.adapter(extra={'dora_scale':np.ones(1,np.float32)})
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1>')
 def test_11_duplicate_alias_rejected(self):
  a,b,_=self.adapter();self.adapter(extra={'unet.encoder.proj.lora_A.weight':a,'unet.encoder.proj.lora_B.weight':b})
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1>')
 def test_12_bad_module_name_blocked(self):
  self.adapter();self.manifest['stages']['encoder']['modules'][0]['input_name']='bad';self.write_manifest()
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1>')
 def test_13_bad_rank_and_target_blocked(self):
  self.adapter();self.manifest['dsp']=68;self.write_manifest()
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1>')
 def test_14_fp16(self):
  self.adapter(dtype=np.float16);self.assertTrue(self.plan('<lora:a:1>').metadata['active'])
 def test_15_nonfinite_blocked(self):
  self.adapter(extra={'lora_unet_encoder_proj.lora_down.weight':np.full((2,4),np.nan,np.float32)})
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1>')
 def test_16_generation_named_input_hooks(self):
  path=ROOT/'src/main/assets/sdxl_runtime/phone_generate.py';tree=ast.parse(path.read_text(encoding='utf-8'))
  chosen=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_lora_inputs','_enc_dec_inputs','_dec_entries_from_enc_out']]
  plan=runtime.Plan('cat','',{}, {'encoder':['rin_lora_m0000:=/a.raw'],'decoder':['rin_lora_m0000:=/b.raw']},{})
  ns={'_ACTIVE_LORA_SESSION':plan,'named_input':lambda name,path:name+':='+str(path),'resolve_output':lambda folder,name,legacy_index:(Path(folder)/(name+'.raw'),{})}
  exec(compile(ast.Module(body=chosen,type_ignores=[]),'actual_driver','exec'),ns)
  for base in ['/condition','/uncondition']:
   self.assertIn('rin_lora_m0000:=/a.raw',ns['_enc_dec_inputs'](base,'smp','ts'))
   self.assertIn('rin_lora_m0000:=/b.raw',ns['_dec_entries_from_enc_out'](base,'/out'))
  ns['_ACTIVE_LORA_SESSION']=None;self.assertEqual(len(ns['_enc_dec_inputs']('/c','s','t')),5);self.assertEqual(len(ns['_dec_entries_from_enc_out']('/c','/out')),12)
 def test_17_control_removed_before_clip(self):
  source=(ROOT/'src/main/assets/sdxl_runtime/phone_generate.py').read_text(encoding='utf-8')
  self.assertIn('prompt, neg_prompt = _ACTIVE_LORA_SESSION.prompt, _ACTIVE_LORA_SESSION.negative',source)
  self.assertNotIn('prepare_lora_request(',source)
 def test_18_exact_template_dtype(self):
  self.adapter();self.manifest['stages']['encoder']['dtype']='float16';self.write_manifest()
  with self.assertRaises(runtime.LoraError):self.plan('<lora:a:1>')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();OUT=Path(args.out)
 if OUT.exists():raise SystemExit('Use a fresh test output directory')
 OUT.mkdir(parents=True)
 with (OUT/'tests.log').open('w',encoding='utf-8') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 record={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'scope':'real loader and injected ONNX arithmetic; synthetic adapters, CPU only'}
 (OUT/'result.json').write_text(json.dumps(record,indent=2));print((OUT/'tests.log').read_text());print(record)
 raise SystemExit(0 if result.wasSuccessful() else 1)
