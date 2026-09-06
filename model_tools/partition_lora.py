"""Partition WAI at natural attention boundaries without changing its arithmetic."""
from pathlib import Path
import copy,hashlib,json,math,os,re
import onnx
from onnx import helper
from independent_lora import inject
from lora_template import tensors
FORMAT='rin-wai-lora-partitioned-inputs-v1'

def partitions(model,cuts,capacity=64):
    nodes=list(model.graph.node);cuts=sorted(set(cuts))
    if not cuts or not all(0<c<len(nodes) for c in cuts):raise ValueError('Invalid partition cuts')
    produced={name:i for i,node in enumerate(nodes) for name in node.output}
    initializers={t.name:t for t in model.graph.initializer}
    constants=set(initializers)
    for node in nodes:
        if node.op_type=='Constant' or node.input and all(not n or n in constants for n in node.input):constants.update(node.output)
    inferred=onnx.shape_inference.infer_shapes(model,data_prop=True)
    values={v.name:v for v in list(inferred.graph.value_info)+list(inferred.graph.input)+list(inferred.graph.output)}
    original_inputs={v.name for v in model.graph.input};original_outputs={v.name for v in model.graph.output}
    consumers={}
    for i,node in enumerate(nodes):
        for name in node.input:consumers.setdefault(name,[]).append(i)
    specs=[];all_boundaries=set()
    for start,stop in zip([0]+cuts,cuts+[len(nodes)]):
        selected=set(range(start,stop));pending=[]
        for i in selected:pending.extend(nodes[i].input)
        while pending:
            name=pending.pop();index=produced.get(name)
            if name in constants and index is not None and index not in selected:
                selected.add(index);pending.extend(nodes[index].input)
        generated={x for i in selected for x in nodes[i].output}
        ins=[]
        for i in sorted(selected):
            for name in nodes[i].input:
                if name and name not in generated and name not in initializers and name not in ins:ins.append(name)
        outs=[]
        for i in range(start,stop):
            for name in nodes[i].output:
                if name in original_outputs or name not in constants and any(j>=stop for j in consumers.get(name,[])):
                    if name not in outs:outs.append(name)
        if not outs:raise ValueError('Empty partition output')
        for name in ins+outs:
            v=values.get(name)
            if v is None or v.type.tensor_type.elem_type!=onnx.TensorProto.FLOAT or any(d.dim_value<=0 for d in v.type.tensor_type.shape.dim):raise ValueError('Boundary is not known static float32: '+name)
        all_boundaries.update(ins+outs);specs.append((sorted(selected),ins,outs))
    aliases={n:f'rin_bridge_{i:03d}' for i,n in enumerate(sorted(all_boundaries-original_inputs-original_outputs))}
    def renamed(value):
        v=copy.deepcopy(value);v.name=aliases.get(v.name,v.name);return v
    results=[];offset=0
    for number,(selected,ins,outs) in enumerate(specs):
        chosen=[copy.deepcopy(nodes[i]) for i in selected];used={n for x in chosen for n in x.input}
        for n in chosen:
            for field in [n.input,n.output]:
                for i,value in enumerate(field):field[i]=aliases.get(value,value)
        graph=helper.make_graph(chosen,f'{model.graph.name}_part{number}',[renamed(values[n]) for n in ins],[renamed(values[n]) for n in outs],[copy.deepcopy(t) for n,t in initializers.items() if n in used])
        part=copy.deepcopy(model);part.graph.CopyFrom(graph)
        expanded,extra=inject(part,capacity)
        def newname(name):
            m=re.match(r'^rin_lora_(\d{4})(.*)$',name)
            return f'rin_lora_{int(m.group(1))+offset:04d}'+m.group(2) if m else name
        for node in expanded.graph.node:
            node.name=newname(node.name)
            for field in [node.input,node.output]:
                for i,name in enumerate(field):field[i]=newname(name)
        for collection in [expanded.graph.input,expanded.graph.output,expanded.graph.value_info,expanded.graph.initializer]:
            for x in collection:x.name=newname(x.name)
        for layer in extra['layers']:
            layer['a']=newname(layer['a']);layer['b']=newname(layer['b'])
        offset+=len(extra['layers'])
        extra.update(inputs=[{'name':aliases.get(n,n),'logical_name':n,'shape':[d.dim_value for d in values[n].type.tensor_type.shape.dim]} for n in ins],outputs=[{'name':aliases.get(n,n),'logical_name':n,'shape':[d.dim_value for d in values[n].type.tensor_type.shape.dim]} for n in outs],part_index=number)
        results.append((expanded,extra))
    return results,{'original_inputs':[v.name for v in model.graph.input],'original_outputs':[v.name for v in model.graph.output],'boundary_aliases':aliases}

def prepare_stage(source,output,stage,capacity=64):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new stage directory')
    base=onnx.load(str(source),load_external_data=False)
    for t in base.graph.initializer:
        if t.data_location==onnx.TensorProto.EXTERNAL and math.prod(t.dims)<=4096:
            onnx.external_data_helper.load_external_data_for_tensor(t,str(source.parent))
            t.ClearField('external_data');t.data_location=onnx.TensorProto.DEFAULT
    pattern=r'/down_blocks\.2/attentions\.\d+/Add' if stage=='encoder' else r'/up_blocks\.0/attentions\.\d+/Add'
    cuts=[i+1 for i,n in enumerate(base.graph.node) if re.fullmatch(pattern,n.name)]
    expected=2 if stage=='encoder' else 3
    if len(cuts)!=expected:raise ValueError('Unexpected WAI attention boundaries')
    segments,logical=partitions(base,cuts,capacity);output.mkdir(parents=True)
    for i,(model,spec) in enumerate(segments):
        folder=output/f'part{i}';folder.mkdir()
        refs={v.value for t in tensors(model.graph) for v in t.external_data if v.key=='location'}
        for ref in refs:
            path=source.parent/ref
            if Path(ref).is_absolute() or '..' in Path(ref).parts or not path.is_file():raise ValueError('Unsafe external data reference')
            target=folder/ref;target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target)
        target=folder/'model.onnx';target.write_bytes(model.SerializeToString());onnx.checker.check_model(str(target))
        spec.update(id=f'{stage}_p{i}',context_file=f'{stage}_p{i}.bin',graph_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),weight_bytes=sum(math.prod(t.dims)*2 for t in model.graph.initializer))
        (folder/'manifest.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    logical.update(parts=[spec for _,spec in segments],layers=[layer for _,spec in segments for layer in spec['layers']],rank_capacity=capacity,source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output/'logical.json').write_text(json.dumps(logical,indent=2),encoding='utf-8');return logical
