"""Verify existing compiled partitions, retaining explicit semantic I/O bindings."""
from pathlib import Path
import json,hashlib,time,traceback,os
ROOT=Path(__file__).parent/'run01'
OUT=Path(__file__).parent/'verification02'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def bind(records,actual):
 result=[];used=set()
 for r in records:
  names=[n for n in [r['name'],r['name']+'_permute'] if n in actual]
  if len(names)!=1:raise ValueError('Missing or ambiguous binding: '+r['name'])
  name=names[0];t=actual[name];shape=r['shape'];dims=t['dimensions'];axes=list(range(len(shape)))
  if dims!=shape:
   if len(shape)==4 and dims==[shape[0],shape[2],shape[3],shape[1]]:axes=[0,2,3,1]
   else:raise ValueError('Unsupported boundary layout: '+name+' '+str((shape,dims)))
  if t['dataType'] not in ['QNN_DATATYPE_FLOAT_16','QNN_DATATYPE_FLOAT_32']:raise ValueError('Unexpected dtype: '+name)
  result.append({'logical':r['name'],'name':name,'shape':shape,'qnn_shape':dims,'qnn_dtype':t['dataType'],'to_qnn_axes':axes});used.add(name)
 return result,used

def state(stage,**kw):
 item=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
 (OUT/'status.json').write_text(json.dumps(item,indent=2),encoding='utf-8');print(json.dumps(item),flush=True)

def main():
 if OUT.exists():raise FileExistsError('Verification already submitted; inspect its state')
 OUT.mkdir()
 try:
  template=json.loads((ROOT/'package/lora_template.build.json').read_text())
  assert template['format']=='rin-wai-lora-partitioned-inputs-v1'
  assert template['model_id']=='wai-v170-sm8750-lora-r64-1024-partitioned-v1'
  template.update(complete=False,soc=69,dsp=79)
  modules=set();changes=[];total=0;count=0
  for stage,spec in template['graphs'].items():
   available=set(spec['original_inputs']);shapes={};flattened=[]
   for part in spec['parts']:
    state('VERIFYING',part=part['id'])
    assert part['context_file']==part['id']+'.bin'
    path=ROOT/'package'/part['context_file'];assert path.is_file() and path.stat().st_size>4096
    info=json.loads((ROOT/'logs'/(part['id']+'.context.json')).read_text())['info']
    assert info['socModel']==69 and info['contextMetadata']['info']['dspArch']==79 and len(info['graphs'])==1
    graph=info['graphs'][0]['info'];inputs={x['info']['name']:x['info'] for x in graph['graphInputs']};outputs={x['info']['name']:x['info'] for x in graph['graphOutputs']}
    ib,used=bind(part['inputs'],inputs);ob,used_out=bind(part['outputs'],outputs)
    assert {x['name'] for x in part['inputs']}.issubset(available)
    for x in part['inputs']:
     if x['name'] in shapes:assert shapes[x['name']]==x['shape']
    for layer in part['layers']:
     assert layer['module'] not in modules and layer['capacity']==template['rank_capacity'];modules.add(layer['module'])
     for letter in ['a','b']:
      t=inputs[layer[letter]];assert t['dimensions']==layer[letter+'_shape'] and t['dataType'] in ['QNN_DATATYPE_FLOAT_16','QNN_DATATYPE_FLOAT_32'];used.add(layer[letter])
    assert used==set(inputs),(part['id'],'inputs',set(inputs)-used)
    assert used_out==set(outputs),(part['id'],'outputs',set(outputs)-used_out)
    before=(path.stat().st_size,path.stat().st_mtime_ns);digest=sha(path);assert before==(path.stat().st_size,path.stat().st_mtime_ns)
    part.update(context_bytes=before[0],context_sha256=digest,qnn_graph_name=graph['graphName'],input_bindings=ib,output_bindings=ob,qnn_inputs={k:v['dimensions'] for k,v in inputs.items()},qnn_outputs={k:v['dimensions'] for k,v in outputs.items()})
    changes.extend({'part':part['id'],'logical':x['logical'],'actual':x['name'],'shape':x['shape']} for x in ob if x['name']!=x['logical'])
    available.update(x['name'] for x in part['outputs']);shapes.update({x['name']:x['shape'] for x in part['outputs']});flattened.extend(part['layers']);total+=before[0];count+=1
   assert spec['layers']==flattened and set(spec['original_outputs']).issubset(available)
  template.update(complete=True,interface_binding='exact-or-explicit-permute-v1')
  (OUT/'lora_template.json').write_text(json.dumps(template,indent=2),encoding='utf-8')
  proof={'parts':count,'layers':len(modules),'bytes':total,'compiler_aliases':changes,'soc':69,'dsp':79,'phone_tested':False,'scope':'compiled interfaces, dependency closure and file integrity; no full-model numerical inference'}
  (OUT/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');state('COMPLETE',**proof)
 except Exception as e:state('FAILED',error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
