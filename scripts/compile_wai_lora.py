"""Workstation-only full WAI LoRA model compilation; new output root per run."""
from pathlib import Path
import argparse,os,sys,json,time,subprocess,hashlib,traceback

def status(root,stage,**extra):
    text=json.dumps({'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**extra},indent=2)
    temp=root/'status.pending';temp.write_text(text);os.replace(temp,root/'status.json');print(text,flush=True)

def available_bytes():
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):raise ctypes.WinError()
    return int(memory.available)

def child(root,sdk,stage):
    tmp=root/'tmp'/stage;tmp.mkdir(parents=True,exist_ok=True)
    os.environ.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),QAIRT_TMP_DIR=str(tmp),TMP=str(tmp),TEMP=str(tmp))
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
    import qairt
    converted=qairt.convert(str(root/'graphs'/stage/'model.onnx'),float_precision=16)
    compiled=qairt.compile(converted,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'))
    compiled.save(str(root/'package'/(stage+'.bin')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-root',required=True);parser.add_argument('--sdk-root',required=True);parser.add_argument('--out',required=True);parser.add_argument('--rank',type=int,default=32);parser.add_argument('--child',choices=['encoder','decoder']);args=parser.parse_args()
    root=Path(args.out).resolve();sdk=Path(args.sdk_root).resolve()
    if args.child:child(root,sdk,args.child);return
    if root.exists():raise SystemExit('Existing build directory: inspect status, do not resubmit')
    if available_bytes()<180*1024**3:raise SystemExit('Need 180 GiB free for offline compilation')
    root.mkdir(parents=True);(root/'package').mkdir();(root/'logs').mkdir()
    try:
        status(root,'BUILDING_GRAPH',runner_pid=os.getpid())
        from build_wai_lora_graph import build_one
        manifest={'schema':1,'format':'rin-lora-packed-linear-v1','base_model':'WAI-illustrious-SDXL-v170','resolution':[1024,1024],'rank_capacity':args.rank,'stages':{},'compiled':False}
        for stage in ['encoder','decoder']:manifest['stages'][stage]=build_one(Path(args.model_root)/stage/'model.onnx',root/'graphs'/stage,args.rank)
        manifest['template_id']=hashlib.sha256(json.dumps(manifest['stages'],sort_keys=True).encode()).hexdigest()
        (root/'package/manifest.build.json').write_text(json.dumps(manifest,indent=2))
        env=os.environ.copy();env['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','')
        for stage in ['encoder','decoder']:
            if available_bytes()<160*1024**3:raise RuntimeError('Insufficient free memory before '+stage)
            cmd=[sys.executable,'-B',__file__,'--model-root',args.model_root,'--sdk-root',str(sdk),'--out',str(root),'--child',stage]
            with (root/'logs'/(stage+'.log')).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000|0x4000)
                while proc.poll() is None:
                    status(root,'COMPILING_'+stage.upper(),runner_pid=os.getpid(),child_pid=proc.pid,free_bytes=available_bytes())
                    time.sleep(10)
                if proc.returncode:raise RuntimeError(stage+' compiler returned '+str(proc.returncode))
            path=root/'package'/(stage+'.bin');assert path.is_file() and path.stat().st_size>4096
            meta=root/'logs'/(stage+'.context.json')
            check=subprocess.run([str(sdk/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe'),'--context_binary='+str(path),'--json_file='+str(meta)],env=env,capture_output=True,text=True,timeout=90,creationflags=0x08000000)
            if check.returncode:raise RuntimeError('Invalid compiled context: '+stage)
            info=json.loads(meta.read_text())['info'];assert info['socModel']==69 and info['contextMetadata']['info']['dspArch']==79
            graph=info['graphs'][0]['info'];inputs={t['info']['name']:t['info']['dimensions'] for t in graph['graphInputs']}
            spec=manifest['stages'][stage];assert inputs['rin_lora']==[spec['elements']],inputs
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
            spec.update(context=path.name,context_bytes=path.stat().st_size,context_sha256=digest.hexdigest(),qnn_inputs=inputs)
        manifest['compiled']=True
        (root/'package/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        status(root,'COMPLETE_OFFLINE',template_id=manifest['template_id'],phone_validated=False)
    except Exception as exc:status(root,'FAILED',error=str(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
