"""Workstation compiler for seven-part WAI LoRA; no source-weight changes."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from compile_layer_contexts import free_memory,sha

def state(root,label,**values):
    values.update(stage=label,time=time.strftime('%Y-%m-%d %H:%M:%S'))
    temp=root/'status.pending';temp.write_text(json.dumps(values,indent=2));os.replace(temp,root/'status.json');print(values,flush=True)

def verify_part(root,sdk,part):
    file=root/'package'/part['context_file'];meta=root/'logs'/(part['id']+'.context.json')
    env=os.environ.copy();env['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','')
    process=subprocess.run([str(sdk/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe'),'--context_binary='+str(file),'--json_file='+str(meta)],env=env,capture_output=True,text=True,timeout=120,creationflags=0x08000000)
    if process.returncode:raise RuntimeError('Metadata parser rejected '+part['id'])
    data=json.loads(meta.read_text())['info'];assert data['socModel']==69 and data['contextMetadata']['info']['dspArch']==79
    graph=data['graphs'][0]['info'];inputs={t['info']['name']:t['info']['dimensions'] for t in graph['graphInputs']};outputs={t['info']['name']:t['info']['dimensions'] for t in graph['graphOutputs']}
    expected={t['name'] for t in part['inputs']}
    for layer in part['layers']:
        for letter in ('a','b'):assert inputs[layer[letter]]==layer[letter+'_shape'];expected.add(layer[letter])
    assert set(inputs)==expected and set(outputs)=={t['name'] for t in part['outputs']}
    for records,actual in [(part['inputs'],inputs),(part['outputs'],outputs)]:
        for t in records:
            shape=t['shape'];dims=actual[t['name']]
            if dims!=shape and not (len(shape)==4 and dims==[shape[0],shape[2],shape[3],shape[1]]):raise RuntimeError('Unexpected I/O layout '+t['name'])
    part.update(context_bytes=file.stat().st_size,context_sha256=sha(file),qnn_inputs=inputs,qnn_outputs=outputs,qnn_graph_name=graph['graphName'])
    (root/'logs'/(part['id']+'.verified.json')).write_text(json.dumps(part,indent=2));return part['id']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--sdk',required=True);parser.add_argument('--out',required=True);parser.add_argument('--child');parser.add_argument('--capacity',type=int,default=64);args=parser.parse_args()
    source,sdk,root=map(Path,[args.source,args.sdk,args.out])
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
        template={'schema':1,'format':FORMAT,'complete':False,'model_id':'wai-v170-sm8750-lora-r64-1024-partitioned-v1','base_model':'WAI-illustrious-SDXL-v170','resolution':[1024,1024],'rank_capacity':args.capacity,'graphs':{}}
        for stage in ('encoder','decoder'):template['graphs'][stage]=prepare_stage(source/stage/'model.onnx',root/'graphs'/stage,stage,args.capacity)
        (root/'package/lora_template.build.json').write_text(json.dumps(template,indent=2))
        parts=[p for spec in template['graphs'].values() for p in spec['parts']]
        def work(part):
            if free_memory()<150*1024**3:raise RuntimeError('Low available memory before '+part['id'])
            command=[sys.executable,'-B',__file__,'--source',str(source),'--sdk',str(sdk),'--out',str(root),'--child',part['id']]
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
        template['complete']=True;(root/'package/lora_template.json').write_text(json.dumps(template,indent=2),encoding='utf-8');state(root,'COMPLETE_OFFLINE',parts=completed,phone_validated=False)
    except Exception as exc:state(root,'FAILED',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
