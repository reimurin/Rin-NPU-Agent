"""Executable real loader + ONNX graph contract tests; CPU only."""
from pathlib import Path
import argparse,sys,json,hashlib,tempfile,copy,unittest
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from safetensors.numpy import save_file
from independent_lora import inject,FORMAT
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/main/assets/sdxl_runtime'))
import phone_lora as loader

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='loader_',dir=OUT);self.base=Path(self.tmp.name);self.folder=self.base/'Lora';self.folder.mkdir();self.rng=np.random.default_rng(163)
  self.template={'schema':1,'format':FORMAT,'complete':True,'model_id':'test','base_model':'WAI-illustrious-SDXL-v170','rank_capacity':64,'resolution':[1024,1024],'graphs':{}}
  self.graphs={};self.modules={};self.values={}
  for stage,module in [('encoder','down_blocks.1.attentions.0.proj_in'),('decoder','up_blocks.1.attentions.0.proj_in')]:
   weight=self.rng.normal(0,.1,(16,12)).astype('f');node=h.make_node('MatMul',['sample','W'],['out'],name='/'+module.replace('.attentions.','/attentions.').replace('.proj_in','/proj_in')+'/MatMul')
   model=h.make_model(h.make_graph([node],stage,[h.make_tensor_value_info('sample',T.FLOAT,[1,3,16])],[h.make_tensor_value_info('out',T.FLOAT,[1,3,12])],[nh.from_array(weight,'W')]),opset_imports=[h.make_operatorsetid('',17)]);model.ir_version=9
   graph,spec=inject(model,64);onnx.checker.check_model(graph);self.graphs[stage]=graph;self.modules[stage]=module;self.values[stage]=weight
   path=self.base/loader.TEMPLATE_REL;path.parent.mkdir(parents=True,exist_ok=True);ctx=path.parent/(stage+'.bin');ctx.write_bytes(b'synthetic context used only for loader integrity checks')
   spec.update(context_file=ctx.name,context_bytes=ctx.stat().st_size,context_sha256=loader.sha(ctx));self.template['graphs'][stage]=spec
  (self.base/loader.TEMPLATE_REL).write_text(json.dumps(self.template))
 def tearDown(self):self.tmp.cleanup()
 def adapter(self,name,rank=4,alpha=8,sgm=False):
  data={};expect={}
  for stage,module in self.modules.items():
   alias=next(x for x in self.template['graphs'][stage]['layers'][0]['aliases'] if x.startswith('lora_unet_'+('input_blocks' if stage=='encoder' else 'output_blocks'))) if sgm else 'unet.'+module
   a=self.rng.normal(0,.1,(rank,16)).astype('f');b=self.rng.normal(0,.1,(12,rank)).astype('f')
   data[alias+'.lora_down.weight']=a;data[alias+'.lora_up.weight']=b;data[alias+'.alpha']=np.array(alpha,dtype='f');expect[stage]=(a,b,alpha/rank)
  save_file(data,str(self.folder/(name+'.safetensors')),metadata={'ss_base_model_version':'sdxl_base_v1-0'});return expect
 def session(self,text):return loader.prepare_request(text,'',self.base,log=lambda x:None)[2]
 def execute(self,session,stage,x):
  opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1;s=ort.InferenceSession(self.graphs[stage].SerializeToString(),sess_options=opt,providers=['CPUExecutionProvider'])
  feeds={'sample':x};layer=self.template['graphs'][stage]['layers'][0]
  for letter in ('a','b'):feeds[layer[letter]]=np.fromfile(session.entries[stage][layer[letter]],dtype='<f4').reshape(layer[letter+'_shape'])
  return s.run(None,feeds)[0]
 def test_real_loader_two_adapter_exact_math(self):
  one=self.adapter('角色',rank=4,alpha=8,sgm=True);two=self.adapter('style',rank=2,alpha=1)
  for u,v in [(.8,0),(1.1,-.4),(0,.7),(.8,.3)]:
   session=self.session(f'<lora:角色:{u}>, <lora:style:{v}>')
   for stage in self.graphs:
    x=self.rng.normal(0,.2,(1,3,16)).astype('f');expected=x@self.values[stage]
    for strength,weights in [(u,one[stage]),(v,two[stage])]:
     a,b,factor=weights;expected+=strength*factor*((x@a.T)@b.T)
    np.testing.assert_allclose(self.execute(session,stage,x),expected,rtol=3e-5,atol=2e-6)
 def test_cache_reused(self):
  self.adapter('one');first=self.session('<lora:one:.8>');second=self.session('<lora:one:.8>');self.assertEqual(first.entries,second.entries)
 def test_strength_changes_cache(self):
  self.adapter('one');self.assertNotEqual(self.session('<lora:one:.8>').entries,self.session('<lora:one:1.1>').entries)
 def test_file_change_invalidates_cache(self):
  self.adapter('one');first=self.session('<lora:one:.8>');self.adapter('one');second=self.session('<lora:one:.8>');self.assertNotEqual(first.entries,second.entries)
 def test_zero_does_not_need_component(self):
  prompt,neg,session=loader.prepare_request('cat <lora:missing:0>','',self.base/'notinstalled');self.assertIsNone(session);self.assertEqual(prompt,'cat ')
 def test_unknown_adapter_not_ignored(self):
  save_file({'text_encoder.foo.lora_A.weight':np.ones((4,16),'f'),'text_encoder.foo.lora_B.weight':np.ones((12,4),'f')},str(self.folder/'bad.safetensors'))
  with self.assertRaises(loader.LoraError):self.session('<lora:bad:.8>')
 def test_rank_over_capacity_rejected(self):
  self.adapter('a',rank=40);self.adapter('b',rank=32)
  with self.assertRaises(loader.LoraError):self.session('<lora:a:1><lora:b:1>')
 def test_named_cfg_rows_receive_same_weights(self):
  self.adapter('a');session=self.session('<lora:a:1>');p=self.base/'inputs.txt';p.write_text('sample:=one.raw\nsample:=two.raw\n')
  session.extend_input_list(session.contexts['encoder'],p);first=p.read_text();session.extend_input_list(session.contexts['encoder'],p);self.assertEqual(first,p.read_text())
  lines=p.read_text().splitlines();self.assertEqual(len(lines),2);self.assertEqual(lines[0].split()[1:],lines[1].split()[1:])
 def test_extra_keys_rejected(self):
  self.adapter('a');path=self.folder/'a.safetensors'
  from safetensors.numpy import load_file
  values=load_file(str(path));values['unknown']=np.ones((2,),'f');save_file(values,str(path))
  with self.assertRaises(loader.LoraError):self.session('<lora:a:.8>')
 def test_corrupt_context_rejected(self):
  self.adapter('a');p=self.base/loader.TEMPLATE_REL;ctx=p.parent/'encoder.bin';ctx.write_bytes(b'x'*ctx.stat().st_size)
  with self.assertRaises(loader.LoraError):self.session('<lora:a:.8>')
 def test_original_model_unchanged(self):
  graph=self.graphs['encoder'];before=graph.SerializeToString()
  with self.assertRaises(ValueError):inject(graph,64)
  self.assertEqual(before,graph.SerializeToString())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 (OUT/'result.json').write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'scope':'CPU actual-loader-to-graph; not phone WAI inference'},indent=2));raise SystemExit(0 if result.wasSuccessful() else 1)
