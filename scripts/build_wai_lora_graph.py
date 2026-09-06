"""Packed dynamic linear LoRA branches for actual exported WAI UNet graphs."""
from pathlib import Path
import argparse,copy,hashlib,json,os,re
import onnx
from onnx import helper,TensorProto

def linear_modules(model):
    tensors={x.name:x for x in model.graph.initializer};seen=set();result=[]
    for index,node in enumerate(model.graph.node):
        if node.op_type not in ('MatMul','Gemm') or len(node.input)<2 or node.input[1] not in tensors:continue
        weight=tensors[node.input[1]]
        if len(weight.dims)!=2:continue
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type=='Gemm':
            if attrs.get('transA',0)!=0 or attrs.get('alpha',1.0)!=1.0:raise ValueError('Unsupported Gemm semantics: '+node.name)
            o,i=weight.dims if attrs.get('transB',0) else reversed(weight.dims)
        else:i,o=weight.dims
        module=node.name.rsplit('/',1)[0].strip('/').replace('/','.')
        if not module or not re.fullmatch(r'[A-Za-z0-9_.]+',module):raise ValueError('Cannot identify module: '+node.name)
        if not module.startswith('unet.'):module='unet.'+module
        if module in seen:raise ValueError('Duplicate module: '+module)
        seen.add(module);result.append({'index':index,'module':module,'in_features':int(i),'out_features':int(o),'node':node.name,'weight':node.input[1]})
    return result

def inject(model,rank):
    if rank not in (8,16,32,64):raise ValueError('Unsupported rank capacity')
    modules=linear_modules(model)
    if not modules:raise ValueError('No linear modules')
    existing={x.name for x in list(model.graph.input)+list(model.graph.initializer)}
    if 'rin_lora' in existing or any(n.name.startswith('rin_lora_') for n in model.graph.node):raise ValueError('Graph already instrumented')
    result=copy.deepcopy(model);nodes=[];offset=0;records=[];by_index={x['index']:x for x in modules}
    def const(name,values):
        result.graph.initializer.append(helper.make_tensor(name,TensorProto.INT64,[len(values)],values));return name
    for idx,node in enumerate(result.graph.node):
        if idx not in by_index:nodes.append(copy.deepcopy(node));continue
        spec=dict(by_index[idx]);prefix='rin_lora_'+str(len(records));i,o=spec['in_features'],spec['out_features'];added=[]
        for which,shape in [('a',[i,rank]),('b',[rank,o])]:
            length=shape[0]*shape[1];spec[which+'_offset']=offset;spec[which+'_elements']=length
            start=const(prefix+'_'+which+'_start',[offset]);end=const(prefix+'_'+which+'_end',[offset+length]);axes=const(prefix+'_'+which+'_axis',[0]);shape_name=const(prefix+'_'+which+'_shape',shape)
            added.extend([helper.make_node('Slice',['rin_lora',start,end,axes],[prefix+'_'+which+'_flat'],name=prefix+'_'+which+'_slice'),helper.make_node('Reshape',[prefix+'_'+which+'_flat',shape_name],[prefix+'_'+which],name=prefix+'_'+which+'_reshape')]);offset+=length
        original=copy.deepcopy(node);old_output=original.output[0];original.output[0]=prefix+'_base';nodes.append(original);nodes.extend(added)
        nodes.extend([helper.make_node('MatMul',[node.input[0],prefix+'_a'],[prefix+'_down'],name=prefix+'_down'),helper.make_node('MatMul',[prefix+'_down',prefix+'_b'],[prefix+'_delta'],name=prefix+'_up'),helper.make_node('Add',[prefix+'_base',prefix+'_delta'],[old_output],name=prefix+'_sum')])
        spec.pop('index');spec['rank_capacity']=rank;records.append(spec)
    result.graph.ClearField('node');result.graph.node.extend(nodes)
    result.graph.input.append(helper.make_tensor_value_info('rin_lora',TensorProto.FLOAT,[offset]))
    return result,{'input_name':'rin_lora','elements':offset,'dtype':'float32','rank_capacity':rank,'modules':records}

def build_one(source,out,rank):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Refusing existing graph directory: '+str(out))
    model=onnx.load(str(source),load_external_data=False);modified,spec=inject(model,rank);references=set()
    for tensor in modified.graph.initializer:
        for item in tensor.external_data:
            if item.key=='location':references.add(item.value)
    for name in references:
        src=(source.parent/name).resolve()
        if not src.is_relative_to(source.parent) or not src.is_file():raise ValueError('Missing/unsafe external weight: '+name)
    out.mkdir(parents=True)
    for name in references:
        src=(source.parent/name).resolve();dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);os.link(src,dest)
    target=out/'model.onnx';onnx.save_model(modified,str(target));onnx.checker.check_model(str(target))
    spec.update(source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),graph_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),base_input_names=[x.name for x in model.graph.input])
    (out/'stage.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    print('GRAPH_READY',out,len(spec['modules']),spec['elements']*4,flush=True);return spec

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',required=True);parser.add_argument('--out',required=True);parser.add_argument('--rank',type=int,default=32);args=parser.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('Use a new output directory')
    stages={name:build_one(Path(args.source_root)/name/'model.onnx',out/name,args.rank) for name in ('encoder','decoder')}
    identity=hashlib.sha256(json.dumps(stages,sort_keys=True).encode()).hexdigest()
    manifest={'schema':1,'format':'rin-lora-packed-linear-v1','base_model':'WAI-illustrious-SDXL-v170','resolution':[1024,1024],'rank_capacity':args.rank,'template_id':identity,'scope':'unet linear layers only; unsupported keys reject rather than silently drop','stages':stages,'compiled':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
