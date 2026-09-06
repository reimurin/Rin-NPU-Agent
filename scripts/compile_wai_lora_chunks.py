"""Compile verified small WAI LoRA partitions on workstation, at most two children."""
from pathlib import Path
import argparse,concurrent.futures,ctypes,hashlib,json,os,shutil,subprocess,sys,threading,time,traceback


def memory_available():
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(name,ctypes.c_ulonglong) for name in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return int(m.available)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def write(path,data):
    temp=Path(path).with_suffix('.pending');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(temp,path)


def compatible(a,b):
    if a==b:return True
    return len(a)==len(b)==4 and any([a[i] for i in perm]==b for perm in [(0,2,3,1),(0,3,1,2)])


def child(root,sdk,stage,name):
    tmp=root/'tmp'/name;tmp.mkdir(parents=True,exist_ok=True)
    os.environ.update(QAIRT_SDK_ROOT=str(sdk),QNN_SDK_ROOT=str(sdk),QAIRT_TMP_DIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import qairt
    converted=qairt.convert(str(root/'graphs'/stage/name/'model.onnx'),float_precision=16)
    compiled=qairt.compile(converted,config=qairt.CompileConfig(backend='HTP',soc_details='chipset:SM8750',log_level='info'))
    compiled.save(str(root/'package'/(name+'.bin')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-root',required=True);ap.add_argument('--sdk-root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--rank',type=int,default=32);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--child');ap.add_argument('--stage');args=ap.parse_args()
    root=Path(args.out).resolve();sdk=Path(args.sdk_root).resolve()
    if args.child:child(root,sdk,args.stage,args.child);return
    if root.exists():raise SystemExit('Existing output: inspect status instead of resubmitting')
    if not 1<=args.workers<=2:raise SystemExit('At most two model compilers')
    if memory_available()<200*1024**3:raise SystemExit('Need 200 GiB available before full model compilation')
    if shutil.disk_usage(root.parent).free<50*1024**3:raise SystemExit('Need 50 GiB free temporary space')
    root.mkdir();(root/'package').mkdir();(root/'logs').mkdir()
    running={};lock=threading.Lock();completed=[]
    def status(phase,**extra):
        with lock:pids=dict(running)
        data={'stage':phase,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'runner_pid':os.getpid(),'children':pids,'completed':list(completed),'free_bytes':memory_available(),**extra}
        write(root/'status.json',data);print(json.dumps(data),flush=True)
    try:
        from build_wai_lora_chunks import build_stage,FORMAT
        status('BUILDING_PARTITIONS')
        manifest={'schema':1,'format':FORMAT,'soc':69,'dsp':79,'base_model':'WAI-illustrious-SDXL-v170','resolution':[1024,1024],'rank_capacity':args.rank,'stages':{},'compiled':False}
        for stage in ['encoder','decoder']:
            manifest['stages'][stage]=build_stage(Path(args.model_root)/stage/'model.onnx',root/'graphs'/stage,args.rank,stage)
        manifest['template_id']=hashlib.sha256(json.dumps(manifest['stages'],sort_keys=True).encode()).hexdigest()
        write(root/'package/manifest.build.json',manifest)
        env=os.environ.copy();env['PATH']=str(sdk/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','')
        jobs=[(s,c) for s,v in manifest['stages'].items() for c in v['chunks']]
        def compile_one(stage,chunk):
            name=chunk['id']
            if memory_available()<100*1024**3:raise RuntimeError('Memory reserve too low before '+name)
            command=[sys.executable,'-B',__file__,'--model-root',args.model_root,'--sdk-root',str(sdk),'--out',str(root),'--child',name,'--stage',stage]
            with (root/'logs'/(name+'.log')).open('w',encoding='utf-8') as log:
                proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000|0x4000)
                with lock:running[name]=proc.pid
                code=proc.wait()
                with lock:running.pop(name,None)
                if code:raise RuntimeError(name+' returned '+str(code))
            path=root/'package'/chunk['context'];meta=root/'logs'/(name+'.context.json')
            check=subprocess.run([str(sdk/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe'),'--context_binary='+str(path),'--json_file='+str(meta)],env=env,capture_output=True,text=True,timeout=90,creationflags=0x08000000)
            if check.returncode:raise RuntimeError('Metadata invalid: '+name+' '+check.stderr[-1000:])
            info=json.loads(meta.read_text())['info']
            if info['socModel']!=69 or info['contextMetadata']['info']['dspArch']!=79:raise RuntimeError('Wrong hardware target: '+name)
            if len(info['graphs'])!=1:raise RuntimeError('Unexpected graph count')
            graph=info['graphs'][0]['info'];io={}
            for category,key in [('inputs','graphInputs'),('outputs','graphOutputs')]:
                actual={v['info']['name']:v['info'] for v in graph[key]};expected={v['name']:v for v in chunk[category]}
                if set(actual)!=set(expected):raise RuntimeError('I/O names changed '+name+' '+category)
                io[category]={}
                for tensor,spec in expected.items():
                    shape=actual[tensor]['dimensions']
                    if not compatible(spec['shape'],shape):raise RuntimeError('Unverified layout '+name+' '+tensor+' '+str((spec['shape'],shape)))
                    io[category][tensor]={'shape':shape,'data_type':actual[tensor]['dataType']}
            return {'context_bytes':path.stat().st_size,'context_sha256':digest(path),'qnn_io':io}
        queue=iter(jobs);futures={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in range(args.workers):
                item=next(queue,None)
                if item:futures[pool.submit(compile_one,*item)]=item
            while futures:
                done,_=concurrent.futures.wait(futures,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
                status('COMPILING_PARTITIONS',total=len(jobs))
                for future in done:
                    stage,chunk=futures.pop(future);chunk.update(future.result());completed.append(chunk['id'])
                    write(root/'package/manifest.build.json',manifest)
                    item=next(queue,None)
                    if item:futures[pool.submit(compile_one,*item)]=item
        manifest['compiled']=True;write(root/'package/manifest.json',manifest)
        status('COMPLETE_OFFLINE',template_id=manifest['template_id'],phone_validated=False)
    except Exception as error:
        status('FAILED',error=str(error),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
