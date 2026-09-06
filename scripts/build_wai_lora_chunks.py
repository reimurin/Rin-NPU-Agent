"""Partition real WAI dynamic LoRA graphs without dropping any module or weight.
CPU graph construction only. Large external weights are never rewritten.
"""
from pathlib import Path
import copy,hashlib,json,math,os,re
import onnx
from onnx import helper,TensorProto
from onnx.external_data_helper import load_external_data_for_tensor
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from build_wai_lora_layer_inputs import inject

FORMAT='rin-lora-chunked-v1'


def concrete_shapes(model,source_dir):
    probe=copy.deepcopy(model)
    for tensor in probe.graph.initializer:
        if tensor.data_location==TensorProto.EXTERNAL and math.prod(tensor.dims)<=65536:
            load_external_data_for_tensor(tensor,str(source_dir))
            tensor.data_location=TensorProto.DEFAULT;tensor.ClearField('external_data')
    probe=SymbolicShapeInference.infer_shapes(probe,auto_merge=True,guess_output_rank=False,verbose=0)
    return {v.name:copy.deepcopy(v) for v in list(probe.graph.input)+list(probe.graph.value_info)+list(probe.graph.output)}


def partition(model,rank,source_dir,stage,weight_budget=640*1024*1024,max_modules=64,max_nodes=850):
    shapes=concrete_shapes(model,source_dir)
    expanded,spec=inject(model,rank)
    nodes=list(expanded.graph.node);initializers={t.name:t for t in expanded.graph.initializer}
    shapes.update({v.name:copy.deepcopy(v) for v in expanded.graph.input})
    # Shape of a statically known tensor is a constant, even across graph cuts.
    for n in nodes:
        if n.op_type=='Shape' and n.input[0] in shapes:
            dims=shapes[n.input[0]].type.tensor_type.shape.dim
            if all(d.HasField('dim_value') for d in dims):
                attr={a.name:helper.get_attribute_value(a) for a in n.attribute}
                values=[d.dim_value for d in dims][attr.get('start',0):attr.get('end',len(dims))]
                n.CopyFrom(helper.make_node('Constant',[],list(n.output),name=n.name,
                    value=helper.make_tensor(n.output[0]+'_static',TensorProto.INT64,[len(values)],values)))
    producer={o:i for i,n in enumerate(nodes) for o in n.output}
    consumers={}
    for i,n in enumerate(nodes):
        for value in n.input:
            if value:consumers.setdefault(value,[]).append(i)
    static=set(initializers);shared=set()
    pure={'Constant','Identity','Cast','Concat','Gather','Unsqueeze','Squeeze','Reshape','Slice','Shape','Size','Add','Sub','Mul','Div','Equal','Where','Expand','ConstantOfShape','Transpose','Range'}
    for i,n in enumerate(nodes):
        if n.op_type in pure and all(not v or v in static for v in n.input):
            shared.add(i);static.update(n.output)
    base_inputs=[v.name for v in model.graph.input];base_outputs=[v.name for v in model.graph.output]
    safe={};counter=0
    for n in nodes:
        for value in n.output:
            if not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',value):
                safe[value]='rin_boundary_'+str(counter).zfill(5);counter+=1
    rename=lambda name:safe.get(name,name)
    by_node={m['node']:i for i,m in enumerate(spec['modules'])}
    boundaries=[];start=0;weight=0;mods=0;ops=0;seen_weights=set()
    for i,n in enumerate(nodes):
        if i not in shared:
            ops+=1;mods+=int(n.name in by_node)
            for name in n.input:
                if name in initializers and name not in seen_weights:
                    t=initializers[name];weight+=math.prod(t.dims)*4;seen_weights.add(name)
        is_cut=(bool(re.search(r'/transformer_blocks\.\d+/Add_2$',n.name)) or
                bool(re.search(r'/resnets\.\d+/Div$',n.name)))
        if is_cut and (weight>=weight_budget or mods>=max_modules or ops>=max_nodes):
            boundaries.append((start,i+1));start=i+1;weight=mods=ops=0;seen_weights=set()
    if start<len(nodes):boundaries.append((start,len(nodes)))
    result=[];records=[];covered=[]
    def vi(name):
        if name not in shapes:raise ValueError('Unknown partition tensor shape: '+name)
        value=copy.deepcopy(shapes[name]);value.name=rename(name)
        dims=value.type.tensor_type.shape.dim
        if any(not d.HasField('dim_value') or d.dim_value<=0 for d in dims):raise ValueError('Non-static partition boundary: '+name)
        if value.type.tensor_type.elem_type!=TensorProto.FLOAT:raise ValueError('Non-float execution boundary requires separate handling: '+name)
        return value
    def schema(value):return {'name':value.name,'shape':[d.dim_value for d in value.type.tensor_type.shape.dim],'dtype':'float32'}
    for idx,(begin,end) in enumerate(boundaries):
        main=[i for i in range(begin,end) if i not in shared]
        if not main:continue
        used_nodes=set(main);used_weights=set();inputs=[];outputs=[]
        def need_constant(name):
            if name in initializers:used_weights.add(name);return
            pi=producer.get(name)
            if pi in shared and pi not in used_nodes:
                used_nodes.add(pi)
                for dep in nodes[pi].input:
                    if dep:need_constant(dep)
        for i in main:
            for name in nodes[i].input:
                if not name:continue
                if name in static:need_constant(name)
                elif name not in producer or producer[name]<begin:
                    if name not in inputs:inputs.append(name)
            for name in nodes[i].output:
                if name in base_outputs or any(c>=end for c in consumers.get(name,[])):
                    if name not in outputs:outputs.append(name)
        if not outputs:raise ValueError('Empty chunk outputs')
        newnodes=[]
        for i in sorted(used_nodes):
            n=copy.deepcopy(nodes[i])
            for k,name in enumerate(n.input):n.input[k]=rename(name)
            for k,name in enumerate(n.output):n.output[k]=rename(name)
            newnodes.append(n)
        ins=[vi(name) for name in inputs];outs=[vi(name) for name in outputs]
        chunk=copy.deepcopy(expanded);chunk.graph.ClearField('node');chunk.graph.node.extend(newnodes)
        chunk.graph.ClearField('input');chunk.graph.input.extend(ins)
        chunk.graph.ClearField('output');chunk.graph.output.extend(outs)
        chunk.graph.ClearField('initializer');chunk.graph.initializer.extend(copy.deepcopy(initializers[name]) for name in sorted(used_weights))
        chunk.graph.ClearField('value_info');chunk.graph.name=stage+'_'+str(idx).zfill(2)
        layers=[by_node[nodes[i].name] for i in main if nodes[i].name in by_node];covered.extend(layers)
        record={'id':chunk.graph.name,'context':chunk.graph.name+'.bin','inputs':[schema(v) for v in ins],
                'outputs':[schema(v) for v in outs],'module_indices':layers,'nodes':len(newnodes),
                'source_weight_bytes':sum(math.prod(t.dims)*4 for t in chunk.graph.initializer)}
        result.append(chunk);records.append(record)
    if sorted(covered)!=list(range(len(spec['modules']))):raise ValueError('Partition lost or duplicated LoRA modules')
    spec.update(chunks=records,original_inputs=[schema(vi(v)) for v in base_inputs],original_outputs=[schema(vi(v)) for v in base_outputs])
    return result,spec


def build_stage(source,out,rank,stage):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use new output directory: '+str(out))
    model=onnx.load(str(source),load_external_data=False)
    chunks,spec=partition(model,rank,source.parent,stage);out.mkdir(parents=True)
    for chunk,record in zip(chunks,spec['chunks']):
        folder=out/record['id'];folder.mkdir()
        refs={e.value for t in chunk.graph.initializer for e in t.external_data if e.key=='location'}
        for relative in refs:
            path=(source.parent/relative).resolve()
            if not path.is_relative_to(source.parent) or not path.is_file():raise ValueError('Unsafe external data path')
            target=folder/relative;target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target)
        path=folder/'model.onnx';onnx.save_model(chunk,str(path));onnx.checker.check_model(str(path))
        record['graph_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        print('CHUNK_READY',record['id'],'nodes',record['nodes'],'modules',len(record['module_indices']),
              'raw_weight_MiB',round(record['source_weight_bytes']/1024**2,1),flush=True)
    spec['source_onnx_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'stage.json').write_text(json.dumps(spec,indent=2),encoding='utf-8');return spec
