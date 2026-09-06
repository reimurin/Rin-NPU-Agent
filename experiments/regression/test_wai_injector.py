from pathlib import Path
import argparse,sys,json
import numpy as np,onnx,onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'));from prepare_wai_lora import inject
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--models',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True)
rng=np.random.default_rng(1603);results=[]
for kind in ['MatMul','Gemm','Conv']:
    w=rng.normal(size=(8,6)).astype(np.float32);a=rng.normal(size=(3,8)).astype(np.float32);b=rng.normal(size=(6,3)).astype(np.float32)
    x=rng.normal(size=(1,4,8) if kind=='MatMul' else (2,8) if kind=='Gemm' else (1,8,2,2)).astype(np.float32)
    weight=w if kind=='MatMul' else w.T if kind=='Gemm' else w.T[:,:,None,None]
    attrs={'transB':1} if kind=='Gemm' else {'kernel_shape':[1,1]} if kind=='Conv' else {}
    node=helper.make_node(kind,['x','unet.test.weight'],['y'],name='/test/'+kind,**attrs)
    shape=[1,4,6] if kind=='MatMul' else [2,6] if kind=='Gemm' else [1,6,2,2]
    graph=helper.make_graph([node],'test',[helper.make_tensor_value_info('x',TensorProto.FLOAT,list(x.shape))],[helper.make_tensor_value_info('y',TensorProto.FLOAT,shape)],[numpy_helper.from_array(weight,'unet.test.weight')])
    model=helper.make_model(graph,opset_imports=[helper.make_operatorsetid('',17)]);model.ir_version=9
    mod,spec=inject(model,capacity=16);onnx.checker.check_model(mod)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    run=ort.InferenceSession(mod.SerializeToString(),sess_options=options,providers=['CPUExecutionProvider']);layer=spec['layers'][0]
    for strength in [0,.8,1.1,-.5,0]:
        aa=np.zeros((8,16),np.float32);bb=np.zeros((16,6),np.float32);aa[:,:3]=a.T;bb[:3,:]=b.T*strength*(2/3)
        actual=run.run(None,{'x':x,layer['a']:aa,layer['b']:bb})[0]
        merged=w+(a.T@b.T)*strength*(2/3)
        expected=x@merged if kind!='Conv' else np.einsum('oi,nihw->nohw',merged.T,x)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=4e-6)
        results.append({'op':kind,'strength':strength,'max_error':float(np.max(np.abs(actual-expected)))})
manifest={'schema':1,'model_id':'wai-v170-sm8750-lora-r64-1024-v1','rank_capacity':64,'resolution':[1024,1024],'graphs':{}}
all_modules=[]
for name in ['encoder','decoder']:
    source=Path(args.models)/name/'model.onnx';model=onnx.load(str(source),load_external_data=False);_,spec=inject(model,capacity=64)
    manifest['graphs'][name]=spec;all_modules.extend(x['module'] for x in spec['layers'])
    print('WAI_MAPPING',name,'layers',len(spec['layers']),'dynamic_MiB',sum((x['in_features']+x['out_features'])*64*4 for x in spec['layers'])/2**20)
assert len(set(all_modules))==len(all_modules),'duplicate modules across splits'
(out/'lora_template.uncompiled.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
report={'numeric_cases':len(results),'passed':True,'numeric_results':results,'total_wai_modules':len(all_modules),'scope':'CPU equivalence and real WAI graph metadata only; not full-model NPU proof'}
(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
