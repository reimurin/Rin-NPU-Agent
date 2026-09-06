"""Inject dynamic low-rank inputs into exact WAI ONNX attachment points.
Weights remain external immutable files. No SDK or model weights are committed.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,re,shutil
from pathlib import Path
import onnx
from onnx import TensorProto,helper

SCHEMA=1

def tensors(graph):
    yield from graph.initializer
    for n in graph.node:
        for a in n.attribute:
            if a.HasField('t'):yield a.t
            yield from a.tensors
            if a.HasField('g'):yield from tensors(a.g)
            for g in a.graphs:yield from tensors(g)

def attachment(n,constants,component):
    if n.op_type not in ('MatMul','Gemm','Conv') or len(n.input)<2 or n.input[1] not in constants:return None
    t=constants[n.input[1]];dims=list(t.dims)
    attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
    if n.op_type=='Gemm' and (attrs.get('transA',0)!=0 or attrs.get('alpha',1)!=1):return None
    if n.op_type=='MatMul' and len(dims)!=2:return None
    if n.op_type=='Gemm' and len(dims)!=2:return None
    if n.op_type=='Conv':
        if len(dims)!=4 or dims[2:]!=[1,1] or attrs.get('group',1)!=1 or any(v!=1 for v in attrs.get('strides',[1,1])) or any(attrs.get('pads',[0,0,0,0])):return None
        kind='conv1x1';out_features,in_features=dims[:2]
    elif n.op_type=='Gemm' and attrs.get('transB',0)==1:
        kind='linear';out_features,in_features=dims
    else:
        kind='linear';in_features,out_features=dims
    # Constant names may be folded by ONNX; use the preserved module-qualified node path.
    if t.name.startswith(component+'.') and t.name.endswith('.weight'):
        module=t.name[:-7]
    else:
        path=n.name.strip('/').rsplit('/',1)[0].replace('/','.')
        if not path or path.startswith('onnx') or '/' not in n.name:return None
        module=component+'.'+path
    if module.startswith(component+'.'+component+'.'):module=module[len(component)+1:]
    return dict(module=module,kind=kind,in_features=in_features,out_features=out_features,
                base_node=n.name,base_weight=t.name,base_weight_shape=dims)

def inject(model,component='unet',capacity=64):
    if capacity not in (16,32,64,128):raise ValueError('Unsupported template capacity')
    model=copy.deepcopy(model);constants={t.name:t for t in model.graph.initializer}
    original_inputs=[x.name for x in model.graph.input]
    existing={x.name for x in model.graph.input}|set(constants)|{o for n in model.graph.node for o in n.output}
    layers=[];nodes=[];seen=set()
    for n in model.graph.node:
        spec=attachment(n,constants,component)
        if spec is None:nodes.append(n);continue
        module=spec['module']
        if module in seen:raise ValueError('Repeated module attachment requires explicit handling: '+module)
        seen.add(module);ident='rin_lora_'+str(len(layers)).zfill(4)
        for suffix in ['_a','_b','_base','_low','_delta','_input_nhwc','_delta_nhwc']:
            if ident+suffix in existing:raise ValueError('Name collision')
        spec.update(a=ident+'_a',b=ident+'_b',capacity=capacity)
        inf,outf=spec['in_features'],spec['out_features']
        spec['a_shape']=[inf,capacity];spec['b_shape']=[capacity,outf]
        for name,shape in [(spec['a'],spec['a_shape']),(spec['b'],spec['b_shape'])]:
            model.graph.input.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))
        original=n.output[0];n.output[0]=ident+'_base';nodes.append(n);x=n.input[0]
        if spec['kind']=='conv1x1':
            nodes.append(helper.make_node('Transpose',[x],[ident+'_input_nhwc'],name=ident+'_in_transpose',perm=[0,2,3,1]));x=ident+'_input_nhwc'
        nodes.append(helper.make_node('MatMul',[x,spec['a']],[ident+'_low'],name=ident+'_down'))
        delta=ident+'_delta_nhwc' if spec['kind']=='conv1x1' else ident+'_delta'
        nodes.append(helper.make_node('MatMul',[ident+'_low',spec['b']],[delta],name=ident+'_up'))
        if spec['kind']=='conv1x1':nodes.append(helper.make_node('Transpose',[delta],[ident+'_delta'],name=ident+'_out_transpose',perm=[0,3,1,2]))
        nodes.append(helper.make_node('Add',[ident+'_base',ident+'_delta'],[original],name=ident+'_add'))
        spec['aliases']=[module, 'lora_'+module.replace('.','_')]
        layers.append(spec)
    if not layers:raise ValueError('No supported attachment points')
    del model.graph.node[:];model.graph.node.extend(nodes)
    return model,dict(schema=SCHEMA,component=component,rank_capacity=capacity,original_inputs=original_inputs,layers=layers)

def prepare(source:Path,out:Path,component:str,capacity:int):
    if out.exists():raise FileExistsError(out)
    original=onnx.load(str(source),load_external_data=False)
    patched,spec=inject(original,component,capacity)
    out.mkdir(parents=True)
    for t in tensors(patched.graph):
        if t.data_location!=TensorProto.EXTERNAL:continue
        loc=next((x.value for x in t.external_data if x.key=='location'),None)
        if not loc:raise ValueError('External tensor has no location')
        relative=Path(loc)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe external location')
        src=source.parent/relative;dest=out/relative
        if not src.is_file():raise FileNotFoundError(src)
        if dest.exists():continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,dest)
        except OSError:shutil.copy2(src,dest)
    model_path=out/'model.onnx'
    onnx.save_model(patched,str(model_path))
    onnx.checker.check_model(str(model_path))
    spec['source_onnx_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    spec['model_onnx_sha256']=hashlib.sha256(model_path.read_bytes()).hexdigest()
    spec['dynamic_bytes_fp32']=sum((x['in_features']+x['out_features'])*capacity*4 for x in spec['layers'])
    (out/'graph_lora.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    return spec

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--component',default='unet');ap.add_argument('--capacity',type=int,default=64);a=ap.parse_args()
    s=prepare(Path(a.source),Path(a.out),a.component,a.capacity)
    print(json.dumps({'layers':len(s['layers']),'dynamic_bytes_fp32':s['dynamic_bytes_fp32'],'inputs':len(s['original_inputs'])+2*len(s['layers'])}))
