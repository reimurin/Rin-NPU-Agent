"""Independent dynamic A/B inputs for WAI linear LoRA; no large InputSlice."""
from pathlib import Path
import copy,hashlib,json,os
import onnx
from onnx import helper
from lora_template import linear_targets,tensors
FORMAT='rin-wai-lora-layer-inputs-v1'

def inject(original,capacity=64):
    if capacity not in (8,16,32,64,128):raise ValueError('Unsupported capacity')
    model=copy.deepcopy(original);targets=linear_targets(model)
    names={s for n in model.graph.node for s in list(n.input)+list(n.output)}
    if any(x.startswith('rin_lora_') for x in names):raise ValueError('Graph already has LoRA')
    inputs=[x.name for x in model.graph.input];outputs=[x.name for x in model.graph.output]
    by_index={t['node_index']:(i,t) for i,t in enumerate(targets)};nodes=[];layers=[]
    for index,node in enumerate(model.graph.node):
        pair=by_index.get(index)
        if pair is None:nodes.append(copy.deepcopy(node));continue
        number,t=pair;prefix=f'rin_lora_{number:04d}';old=node.output[0]
        base=copy.deepcopy(node);base.output[0]=prefix+'_base';nodes.append(base)
        for label,shape in [('a',[t['in_features'],capacity]),('b',[capacity,t['out_features']])]:
            model.graph.input.append(helper.make_tensor_value_info(prefix+'_'+label,t['dtype'],shape))
        nodes.extend([helper.make_node('MatMul',[t['input'],prefix+'_a'],[prefix+'_down'],name=prefix+'_down'),helper.make_node('MatMul',[prefix+'_down',prefix+'_b'],[prefix+'_delta'],name=prefix+'_up')])
        delta=prefix+'_delta'
        if t['gemm_alpha']!=1.0:
            factor=prefix+'_factor';model.graph.initializer.append(helper.make_tensor(factor,t['dtype'],[],[t['gemm_alpha']]))
            nodes.append(helper.make_node('Mul',[delta,factor],[prefix+'_scaled'],name=prefix+'_scale'));delta=prefix+'_scaled'
        nodes.append(helper.make_node('Add',[prefix+'_base',delta],[old],name=prefix+'_sum'))
        layer={k:v for k,v in t.items() if k not in ('node_index','dtype','input','output')}
        layer.update(a=prefix+'_a',b=prefix+'_b',a_shape=[t['in_features'],capacity],b_shape=[capacity,t['out_features']],capacity=capacity)
        layers.append(layer)
    model.graph.ClearField('node');model.graph.node.extend(nodes)
    return model,dict(format=FORMAT,io_dtype='float32',rank_capacity=capacity,original_inputs=inputs,original_outputs=outputs,layers=layers)

def prepare(source,output,capacity=64):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Graph output exists')
    original=onnx.load(str(source),load_external_data=False)
    refs={v.value for t in tensors(original.graph) for v in t.external_data if v.key=='location'}
    for ref in refs:
        path=source.parent/ref
        if Path(ref).is_absolute() or '..' in Path(ref).parts or not path.is_file():raise ValueError('Unsafe/missing external reference')
    model,spec=inject(original,capacity);output.parent.mkdir(parents=True,exist_ok=True)
    for ref in refs:
        src=source.parent/ref;target=output.parent/ref;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if not target.samefile(src):raise FileExistsError('External target collision')
        else:os.link(src,target)
    output.write_bytes(model.SerializeToString());onnx.checker.check_model(str(output))
    spec.update(source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.manifest.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    return spec
