from pathlib import Path
import argparse,hashlib,json,time,os,re,traceback,psutil
ap=argparse.ArgumentParser();ap.add_argument('--receive',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();receive=Path(a.receive);out=Path(a.out)
if (out/'status.json').exists():raise RuntimeError('Task already submitted')
out.mkdir(parents=True,exist_ok=True)
def state(stage,**fields):
 p=out/'status.next.json';p.write_text(json.dumps(dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**fields),indent=2));os.replace(p,out/'status.json');print(stage,fields,flush=True)
def sha(file):
 h=hashlib.sha256()
 with file.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ['onnx_path','source_onnx','source_path','tmp_dir','source_dir']}
 if isinstance(value,list):return [clean(v) for v in value]
 return value
try:
 manifest=clean(json.loads((receive/'metadata/lora_template.json').read_text()));assert manifest['complete'] and manifest['soc']==69 and manifest['dsp']==79
 tag='wai-lora-sm8750-r64-partitioned-v1';prefix='https://github.com/reimurin/Rin-NPU-Agent/releases/download/'+tag+'/'
 specs=[p for s in manifest['graphs'].values() for p in s['parts']];descriptor={'schema':1,'model_id':manifest['model_id'],'soc':69,'dsp':79,'minimum_app_version_code':20,'files':[]};verified=[]
 for index,spec in enumerate(specs):
  file=receive/spec['context_file'];deadline=time.monotonic()+7200
  while True:
   receiver_pid=int((receive/'receiver.pid').read_text())
   active=psutil.pid_exists(receiver_pid) and psutil.Process(receiver_pid).name().lower()=='croc.exe'
   ready=file.is_file() and file.stat().st_size==spec['context_bytes'] and (not active or index+1<len(specs) and (receive/specs[index+1]['context_file']).exists())
   if ready:
    state('VERIFY_FILE',file=file.name);actual=sha(file)
    if actual==spec['context_sha256']:break
    if not active:raise RuntimeError('Received model hash mismatch: '+file.name)
   if not active:raise RuntimeError('Receiver exited before model complete: '+file.name)
   if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for existing transfer; no restart performed')
   state('WAIT_EXISTING_TRANSFER',file=file.name);time.sleep(20)
  verified.append({'file':file.name,'bytes':spec['context_bytes'],'sha256':actual})
  state('PACKAGING',file=file.name)
  parts=[]
  with file.open('rb') as source:
   number=0
   while True:
    data=source.read(512*1024*1024)
    if not data:break
    name=file.name+'.'+str(number).zfill(3);target=out/name
    if target.exists():raise RuntimeError('Shard exists unexpectedly: '+name)
    target.write_bytes(data);digest=hashlib.sha256(data).hexdigest();parts.append({'name':name,'url':prefix+name,'bytes':len(data),'sha256':digest});number+=1
  assert sum(p['bytes'] for p in parts)==spec['context_bytes']
  descriptor['files'].append({'name':file.name,'bytes':spec['context_bytes'],'sha256':actual,'parts':parts})
  (out/'verified-models.json').write_text(json.dumps(verified,indent=2))
 data=json.dumps(manifest,ensure_ascii=False,indent=2).encode();name='lora_template.json.000';(out/name).write_bytes(data);digest=hashlib.sha256(data).hexdigest()
 descriptor['files'].append({'name':'lora_template.json','bytes':len(data),'sha256':digest,'parts':[{'name':name,'url':prefix+name,'bytes':len(data),'sha256':digest}]})
 (out/'component.json').write_text(json.dumps(descriptor,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=[]
 for item in descriptor['files']:
  for part in item['parts']:lines.append(part['sha256']+'  '+part['name'])
 lines.append(sha(out/'component.json')+'  component.json');(out/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
 state('COMPLETE',files=len(specs),shards=len(lines)-1,bytes=sum(s['context_bytes'] for s in specs),descriptor_sha256=sha(out/'component.json'))
except Exception as e:state('FAILED_OR_UNCONFIRMED',error=str(e),traceback=traceback.format_exc());raise
