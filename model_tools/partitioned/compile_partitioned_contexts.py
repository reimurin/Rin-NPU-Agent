"""Workstation compiler for seven-part WAI LoRA; no source-weight changes."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from compile_layer_contexts import free_memory,sha

def state(root,label,**values):
    values.update(stage=label,time=time.strftime('%Y-%m-%d %H:%M:%S'))
    temp=root/'status.pending';temp.write_text(json.dumps(values,indent=2));os.replace(temp,root/'status.json');print(values,flush=True)

def bind(records,actual):
    result=[];used=set()
    for record in records:
        logical=record['name'];candidates=[name for name in (logical,logical+'_permute') if name in actual]
        if len(candidates)!=1:raise ValueError('Missing or ambiguous semantic binding: '+logical)
        name=candidates[0];tensor=actual[name];shape=record['shape'];dims=tensor['dimensions'];axes=list(range(len(shape)))
        if dims!=shape:
            if len(shape)==4 and dims==[shape[0],shape[2],shape[3],shape[1]]:axes=[0,2,3,1]
            else:raise ValueError('Unsupported boundary layout: '+name+' '+str((shape,dims)))
        if tensor.get('dataType') not in ('QNN_DATATYPE_FLOAT_16','QNN_DATATYPE_FLOAT_32'):raise ValueError('Unexpected dtype: '+name)
        result.append({'logical':logical,'name':name,'shape':shape,'qnn_shape':dims,'qnn_dtype':tensor['dataType'],'to_qnn_axes':axes});used.add(name)
    return result,used

def verify_part(root,sdk,part):
    file=root/'package'/part['context_file'];meta=root/'logs'/(part['id']+'.context.json')
    env=os.environ.copy();env['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','')
    process=subprocess.run([str(sdk/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe'),'--context_binary='+str(file),'--json_file='+str(meta)],env=env,capture_output=True,text=True,timeout=120,creationflags=0x08000000)
    if process.returncode:raise RuntimeError('Metadata parser rejected '+part['id'])
    data=json.loads(meta.read_text())['info']
    if data['socModel']!=69 or data['contextMetadata']['info']['dspArch']!=79 or len(data.get('graphs',[]))!=1:raise RuntimeError('Wrong target metadata: '+part['id'])
    graph=data['graphs'][0]['info']
    inputs={t['info']['name']:t['info'] for t in graph['graphInputs']}
    outputs={t['info']['name']:t['info'] for t in graph['graphOutputs']}
    input_bindings,used_inputs=bind(part['inputs'],inputs)
    output_bindings,used_outputs=bind(part['outputs'],outputs)
    for layer in part['layers']:
        for letter in ('a','b'):
            name=layer[letter];tensor=inputs.get(name)
            if tensor is None or tensor['dimensions']!=layer[letter+'_shape'] or tensor.get('dataType') not in ('QNN_DATATYPE_FLOAT_16','QNN_DATATYPE_FLOAT_32'):
                raise RuntimeError('Compiled LoRA input mismatch: '+part['id']+' '+name)
            used_inputs.add(name)
    if used_inputs!=set(inputs):raise RuntimeError('Unbound QNN inputs '+part['id']+': '+str(sorted(set(inputs)-used_inputs)))
    if used_outputs!=set(outputs):raise RuntimeError('Unbound QNN outputs '+part['id']+': '+str(sorted(set(outputs)-used_outputs)))
    part.update(
        context_bytes=file.stat().st_size,
        context_sha256=sha(file),
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        qnn_inputs={k:v['dimensions'] for k,v in inputs.items()},
        qnn_outputs={k:v['dimensions'] for k,v in outputs.items()},
        qnn_graph_name=graph['graphName'],
    )
    (root/'logs'/(part['id']+'.verified.json')).write_text(json.dumps(part,indent=2),encoding='utf-8')
    return part['id']

def parse_resolution(raw):
    try:width,height=(int(x) for x in raw.lower().split('x',1))
    except Exception as exc:raise SystemExit('Invalid --resolution; expected WxH') from exc
    if width<64 or height<64 or width>8192 or height>8192 or width%8 or height%8:raise SystemExit('Unsupported --resolution')
    return width,height

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--sdk',required=True);parser.add_argument('--out',required=True);parser.add_argument('--child');parser.add_argument('--capacity',type=int,default=64);parser.add_argument('--resolution',default='1024x1024');args=parser.parse_args()
    source,sdk,root=map(Path,[args.source,args.sdk,args.out]);width,height=parse_resolution(args.resolution)
    if args.child:
        part=args.child;stage,number=part.split('_p');temp=root/'tmp'/part;temp.mkdir(parents=True,exist_ok=True)
        os.environ.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),QAIRT_TMP_DIR=str(temp),TEMP=str(temp),TMP=str(temp))
        os.environ['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+os.environ.get('PATH','')
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
        import qairt
        model=qairt.convert(str(root/'graphs'/stage/('part'+number)/'model.onnx'),float_precision=16)
        compiled=qairt.compile(model,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'));compiled.save(str(root/'package'/(part+'.bin')));return
    if root.exists():raise SystemExit('Existing model job: inspect status instead of resubmitting')
    if free_memory()<220*1024**3:raise SystemExit('Need 220 GiB free for two model workers')
    root.mkdir(parents=True);(root/'package').mkdir();(root/'logs').mkdir()
    try:
        from partition_lora import prepare_stage,FORMAT
        state(root,'PARTITIONING',runner_pid=os.getpid())
        model_id='wai-v170-sm8750-lora-r64-1024-partitioned-v1' if (width,height)==(1024,1024) else f'wai-v170-sm8750-lora-r64-{width}x{height}-partitioned-v1'
        template={'schema':1,'format':FORMAT,'complete':False,'model_id':model_id,'base_model':'WAI-illustrious-SDXL-v170','resolution':[width,height],'rank_capacity':args.capacity,'graphs':{}}
        for stage in ('encoder','decoder'):template['graphs'][stage]=prepare_stage(source/stage/'model.onnx',root/'graphs'/stage,stage,args.capacity)
        (root/'package/lora_template.build.json').write_text(json.dumps(template,indent=2))
        parts=[p for spec in template['graphs'].values() for p in spec['parts']]
        def work(part):
            if free_memory()<150*1024**3:raise RuntimeError('Low available memory before '+part['id'])
            command=[sys.executable,'-B',__file__,'--source',str(source),'--sdk',str(sdk),'--out',str(root),'--child',part['id'],'--resolution',args.resolution]
            with (root/'logs'/(part['id']+'.log')).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000|0x4000)
                (root/'logs'/(part['id']+'.pid')).write_text(str(proc.pid));rc=proc.wait()
                if rc:raise RuntimeError(part['id']+' compiler returned '+str(rc))
            return verify_part(root,sdk,part)
        completed=[];errors=[];state(root,'COMPILING_PARTS',runner_pid=os.getpid(),workers=2,parts=[p['id'] for p in parts])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(work,p):p['id'] for p in parts}
            for future in as_completed(futures):
                try:completed.append(future.result())
                except Exception as exc:errors.append({'part':futures[future],'error':str(exc)});traceback.print_exc()
                state(root,'COMPILING_PARTS',runner_pid=os.getpid(),completed=completed,errors=errors,free_bytes=free_memory())
        if errors:raise RuntimeError('Failed model parts: '+json.dumps(errors))
        for spec in template['graphs'].values():
            available=set(spec['original_inputs'])
            for part in spec['parts']:
                assert {x['name'] for x in part['inputs']}.issubset(available);available.update(x['name'] for x in part['outputs'])
            assert set(spec['original_outputs']).issubset(available)
        template.update(complete=True,soc=69,dsp=79,interface_binding='exact-or-explicit-permute-v1')
        (root/'package/lora_template.json').write_text(json.dumps(template,indent=2),encoding='utf-8');state(root,'COMPLETE_OFFLINE',parts=completed,phone_validated=False)
    except Exception as exc:state(root,'FAILED',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
