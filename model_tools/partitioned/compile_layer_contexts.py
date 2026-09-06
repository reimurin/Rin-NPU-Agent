"""Full WAI independent-input LoRA compilation, workstation-only."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,sys,time,traceback
from ctypes import wintypes

def free_memory():
    class Memory(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
    data=Memory();data.length=ctypes.sizeof(data)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):raise ctypes.WinError()
    return int(data.available)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def state(out,label,**kw):
    tmp=out/'status.pending';tmp.write_text(json.dumps({'stage':label,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw},indent=2));os.replace(tmp,out/'status.json')
    print(label,kw,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--sdk',required=True);p.add_argument('--out',required=True);p.add_argument('--capacity',type=int,default=64);p.add_argument('--child',choices=['encoder','decoder']);a=p.parse_args()
    source,sdk,out=map(Path,[a.source,a.sdk,a.out]);tmp=out/'tmp'/(a.child or 'driver')
    os.environ.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),QAIRT_TMP_DIR=str(tmp),TEMP=str(tmp),TMP=str(tmp))
    os.environ['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+os.environ.get('PATH','')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
    if a.child:
        tmp.mkdir(parents=True,exist_ok=True);import qairt
        graph=out/'graphs'/a.child/'model.onnx';converted=qairt.convert(str(graph),float_precision=16)
        compiled=qairt.compile(converted,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'))
        compiled.save(str(out/'package'/(a.child+'.bin')));return
    if out.exists():raise SystemExit('Previous job directory exists; inspect before retry')
    if free_memory()<180*1024**3:raise SystemExit('Insufficient free memory before full-model compilation')
    out.mkdir(parents=True);(out/'package').mkdir();(out/'logs').mkdir();tmp.mkdir(parents=True)
    try:
        from independent_lora import prepare,FORMAT
        template={'schema':1,'format':FORMAT,'complete':False,'model_id':f'wai-v170-sm8750-lora-r{a.capacity}-1024-v1','base_model':'WAI-illustrious-SDXL-v170','resolution':[1024,1024],'rank_capacity':a.capacity,'graphs':{}}
        state(out,'PREPARING_GRAPHS',runner_pid=os.getpid())
        for stage in ['encoder','decoder']:template['graphs'][stage]=prepare(source/stage/'model.onnx',out/'graphs'/stage/'model.onnx',a.capacity)
        (out/'package/lora_template.build.json').write_text(json.dumps(template,indent=2))
        for stage in ['encoder','decoder']:
            if free_memory()<160*1024**3:raise RuntimeError('Not enough free memory to start '+stage)
            command=[sys.executable,'-B',__file__,'--source',str(source),'--sdk',str(sdk),'--out',str(out),'--child',stage]
            with (out/'logs'/(stage+'.log')).open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000|0x4000)
                while child.poll() is None:
                    state(out,'COMPILING_'+stage.upper(),runner_pid=os.getpid(),child_pid=child.pid,free_bytes=free_memory());time.sleep(10)
                if child.returncode:raise RuntimeError(stage+' compile failed rc='+str(child.returncode))
            file=out/'package'/(stage+'.bin');meta=out/'logs'/(stage+'.json')
            result=subprocess.run([str(sdk/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe'),'--context_binary='+str(file),'--json_file='+str(meta)],capture_output=True,text=True,timeout=120,creationflags=0x08000000)
            if result.returncode:raise RuntimeError('Official metadata parser rejected '+stage)
            data=json.loads(meta.read_text())['info'];assert data['socModel']==69 and data['contextMetadata']['info']['dspArch']==79
            graph=data['graphs'][0]['info'];inputs={x['info']['name']:x['info']['dimensions'] for x in graph['graphInputs']};spec=template['graphs'][stage]
            expected=set(spec['original_inputs'])
            for layer in spec['layers']:
                for field in ['a','b']:
                    assert inputs[layer[field]]==layer[field+'_shape'],(stage,layer[field],inputs.get(layer[field]));expected.add(layer[field])
            assert set(inputs)==expected,(len(inputs),len(expected))
            spec.update(context_file=file.name,context_bytes=file.stat().st_size,context_sha256=sha(file),qnn_graph_name=graph['graphName'])
            state(out,'VERIFIED_'+stage.upper(),inputs=len(inputs),context_bytes=file.stat().st_size)
        template['complete']=True;(out/'package/lora_template.json').write_text(json.dumps(template,indent=2),encoding='utf-8')
        state(out,'COMPLETE_OFFLINE',model_id=template['model_id'],phone_validated=False)
    except Exception as exc:state(out,'FAILED',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
