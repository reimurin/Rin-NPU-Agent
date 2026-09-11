from pathlib import Path
import sys,os,json,copy,unittest,collections,struct,urllib.request,subprocess
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
from lora_template import inject,linear_targets,module_aliases
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(163)
def graph(gemm=False):
 xshape=[3,16] if gemm else [1,3,16]
 weight=rng.normal(0,.2,(16,12)).astype(np.float32)
 nodes=[helper.make_node('Gemm' if gemm else 'MatMul',['sample','W'],['output'],name='/down_blocks.1/attentions.0/transformer_blocks.0/attn1/to_q/'+('Gemm' if gemm else 'MatMul'),**({'transB':1,'alpha':.5} if gemm else {}))]
 g=helper.make_graph(nodes,'packed_test',[helper.make_tensor_value_info('sample',TensorProto.FLOAT,xshape)],[helper.make_tensor_value_info('output',TensorProto.FLOAT,xshape[:-1]+[12])],[numpy_helper.from_array(weight.T.copy() if gemm else weight,'W')])
 m=helper.make_model(g,opset_imports=[helper.make_operatorsetid('',17)]);m.ir_version=9
 return m,weight,xshape
class Tests(unittest.TestCase):
 def test_alias_diffusers(self):self.assertIn('lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q',module_aliases('down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q'))
 def test_alias_sgm(self):self.assertIn('lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q',module_aliases('down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q'))
 def test_alias_up(self):self.assertIn('lora_unet_output_blocks_4_1_proj_in',module_aliases('up_blocks.1.attentions.1.proj_in'))
 def test_alias_resnet(self):self.assertIn('lora_unet_input_blocks_4_0_emb_layers_1',module_aliases('down_blocks.1.resnets.0.time_emb_proj'))
 def test_repeat_injection_rejected(self):
  m,_,_=graph();m,_=inject(m,8)
  with self.assertRaises(ValueError):inject(m,8)
 def test_invalid_rank(self):
  m,_,_=graph()
  with self.assertRaises(ValueError):inject(m,3)
 def test_matmul_and_gemm_exact_two_adapters(self):
  for gemm in [False,True]:
   model,weight,shape=graph(gemm);raw=[t.SerializeToString() for t in model.graph.initializer];m,info=inject(model,8)
   onnx.checker.check_model(m);self.assertEqual([t.SerializeToString() for t in m.graph.initializer[:len(raw)]],raw)
   opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
   sess=ort.InferenceSession(m.SerializeToString(),sess_options=opt,providers=['CPUExecutionProvider']);x=rng.normal(size=shape).astype(np.float32)
   a1=rng.normal(0,.1,(16,3)).astype(np.float32);b1=rng.normal(0,.1,(3,12)).astype(np.float32);a2=rng.normal(0,.1,(16,2)).astype(np.float32);b2=rng.normal(0,.1,(2,12)).astype(np.float32)
   factor=.5 if gemm else 1;layer=info['layers'][0]
   for w1,w2 in [(0,0),(.8,0),(1.1,-.4),(0,.7),(0,0)]:
    bank=np.zeros(info['bank_shape'],np.float32);A=bank[0,:16*8].reshape(16,8);B=bank[0,16*8:].reshape(8,12)
    A[:,:3]=a1;A[:,3:5]=a2;B[:3]=b1*w1*factor;B[3:5]=b2*w2*factor
    actual=sess.run(None,{'sample':x,'rin_lora_bank':bank})[0];expected=factor*(x@weight)+factor*(w1*((x@a1)@b1)+w2*((x@a2)@b2))
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
   self.assertEqual(info['original_inputs'],['sample']);self.assertEqual(info['original_outputs'],['output'])
 def test_mixed_schema_duplicate_alias(self):
  m,_,_=graph();m.graph.node.add().CopyFrom(m.graph.node[0]);m.graph.node[-1].output[0]='other'
  with self.assertRaises(ValueError):linear_targets(m)
report={'scope':'metadata and CPU test; not phone or full WAI image validation'}
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));report['tests']={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}
if not result.wasSuccessful():raise SystemExit(1)
full={}
for stage in ['encoder','decoder']:
 path=root/'toolchain/sdxl-build/wai-v170-sm8750-r1/onnx_unet_split'/stage/'model.onnx'
 m=onnx.load(str(path),load_external_data=False);targets=linear_targets(m);size=sum((t['in_features']+t['out_features'])*64*4 for t in targets)
 full[stage]={'layers':len(targets),'bank_bytes_float32':size,'aliases':{a:t['module'] for t in targets for a in t['aliases']}}
 print('FULL',stage,len(targets),'bank_mib',size/2**20,flush=True)
report['full']={k:{a:b for a,b in v.items() if a!='aliases'} for k,v in full.items()}
for repo in ['nerijs/pixel-art-xl','ostris/watercolor_style_lora_sdxl']:
 try:
  with urllib.request.urlopen('https://huggingface.co/api/models/'+repo,timeout=20) as f:data=json.load(f)
  files=[x['rfilename'] for x in data['siblings'] if x['rfilename'].endswith('.safetensors')]
  if not files:continue
  file=files[0];rev=data['sha'];url=f'https://huggingface.co/{repo}/resolve/{rev}/{file}'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1048575'}),timeout=25) as f:
   n=struct.unpack('<Q',f.read(8))[0];assert n<8*1024*1024;header=json.loads(f.read(n))
  prefixes=collections.Counter();ranks=collections.Counter();unknown=[];matched=0;aliases={a for x in full.values() for a in x['aliases']}
  for k,v in header.items():
   if not k.endswith('.lora_down.weight'):continue
   module=k[:-17];prefixes[module.split('_')[1] if module.startswith('lora_') else module.split('.')[0]]+=1;ranks[str(v['shape'])]+=1
   if module in aliases:matched+=1
   else:unknown.append(module)
  entry={'repo':repo,'revision':rev,'filename':file,'prefixes':dict(prefixes),'shapes':dict(ranks),'matched_modules':matched,'unmatched_modules':unknown,'metadata':{k:v for k,v in header.get('__metadata__',{}).items() if k in ['ss_base_model_version','ss_network_dim','ss_network_alpha']}}
  report.setdefault('real_loras',[]).append(entry)
  (out/(repo.split('/')[-1]+'.header.json')).write_text(json.dumps(header),encoding='utf-8')
  print('REAL_LORA',json.dumps(entry),flush=True)
 except Exception as e:print('LORA_LOOKUP_ERROR',repo,type(e).__name__,str(e),flush=True)
small,_,_=graph();small,manifest=inject(small,8);onnx.save(small,out/'packed.onnx');(out/'packed.manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(out/'status.txt').write_text('COMPLETE')
