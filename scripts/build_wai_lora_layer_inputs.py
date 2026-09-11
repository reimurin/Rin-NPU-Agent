"""Per-layer dynamic LoRA banks avoid the giant InputSlice VTCM allocation.

Reuses the numerically checked linear projection injector, then partitions its
weight input without changing base weights or the rank-concatenation arithmetic.
"""
from pathlib import Path
import hashlib,json,os
import onnx
from onnx import helper,TensorProto
from build_wai_lora_graph import inject as packed_inject

FORMAT='rin-lora-module-banks-v1'

def inject(model,rank):
    result,spec=packed_inject(model,rank)
    original=list(result.graph.input)
    result.graph.ClearField('input')
    result.graph.input.extend(x for x in original if x.name!='rin_lora')
    constants={t.name:t for t in result.graph.initializer}
    nodes={n.name:n for n in result.graph.node}
    names=[]
    for index,layer in enumerate(spec['modules']):
        prefix='rin_lora_'+str(index);input_name='rin_lora_m'+str(index).zfill(4)
        ae,be=layer['a_elements'],layer['b_elements'];total=ae+be
        if total*2>2*1024*1024:raise ValueError('Per-layer FP16 bank exceeds 2 MiB: '+layer['module'])
        layer['input_name']=input_name;layer['input_elements']=total
        for letter,start,end in [('a',0,ae),('b',ae,total)]:
            for suffix,value in [('start',start),('end',end)]:
                name=prefix+'_'+letter+'_'+suffix
                constants[name].CopyFrom(helper.make_tensor(name,TensorProto.INT64,[1],[value]))
            nodes[prefix+'_'+letter+'_slice'].input[0]=input_name
        result.graph.input.append(helper.make_tensor_value_info(input_name,TensorProto.FLOAT,[total]));names.append(input_name)
    spec['input_layout']='per_module';spec['input_names']=names
    return result,spec

def build_one(source,out,rank):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a new graph directory: '+str(out))
    model=onnx.load(str(source),load_external_data=False);modified,spec=inject(model,rank)
    refs={item.value for tensor in modified.graph.initializer for item in tensor.external_data if item.key=='location'}
    for name in refs:
        src=(source.parent/name).resolve()
        if not src.is_relative_to(source.parent) or not src.is_file():raise ValueError('Invalid external tensor: '+name)
    out.mkdir(parents=True)
    for name in refs:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(source.parent/name,target)
    path=out/'model.onnx';onnx.save_model(modified,str(path));onnx.checker.check_model(str(path))
    spec.update(source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_input_names=[x.name for x in model.graph.input])
    (out/'stage.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    print('LAYER_BANK_GRAPH_READY',out,'modules',len(spec['modules']),'total_fp32_bytes',spec['elements']*4,'max_bank_fp32_bytes',max(m['input_elements']*4 for m in spec['modules']),flush=True)
    return spec
