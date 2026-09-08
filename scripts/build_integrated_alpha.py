from pathlib import Path
import argparse,os,subprocess,sys,shutil,json,hashlib,re,zipfile,traceback
ap=argparse.ArgumentParser();ap.add_argument('--toolchain',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();tool=Path(args.toolchain);out=Path(args.out);app=Path(__file__).resolve().parents[1]
gradle_text=(app/'build.gradle').read_text(encoding='utf-8')
version=re.search(r"versionName\s+['\"]([^'\"]+)['\"]",gradle_text).group(1)
code=int(re.search(r"versionCode\s+(\d+)",gradle_text).group(1))
if (out/'result.json').exists():raise SystemExit('Prior build exists; choose a new output directory')
out.mkdir(parents=True,exist_ok=True)
def state(s):(out/'status.txt').write_text(s,encoding='utf-8');print(s,flush=True)
env=os.environ.copy();env.update(JAVA_HOME=str(tool/'jdk-17'),ANDROID_HOME=str(tool/'android-sdk'),ANDROID_SDK_ROOT=str(tool/'android-sdk'),ANDROID_USER_HOME=str(tool/'android-user-home'),GRADLE_USER_HOME=str(tool/'gradle-home'),TEMP=str(out/'tmp'),TMP=str(out/'tmp'),PYTHONDONTWRITEBYTECODE='1')
(out/'tmp').mkdir(exist_ok=True)
env['RIN_RENDER_DIR']=str(out/'ui')
env['JAVA_TOOL_OPTIONS']='-Djava.io.tmpdir='+str(out/'tmp')
env['PATH']=str(tool/'jdk-17/bin')+os.pathsep+env.get('PATH','')
flags=0x08000000|0x4000
log=out/'build.log'
probe_asset=app/'src/main/assets/sdxl_runtime/qnn_probe/libCalculator_skel.so'
probe_sha='5f5f0ab9f1264c86f0fd8cdbd7873bb731bdcece6f7e91624d1a5ae8943e578e'
def run(label,cmd,cwd=app):
 state(label)
 with log.open('a',encoding='utf-8') as f:
  p=subprocess.run(list(map(str,cmd)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=flags,timeout=900)
 if p.returncode:raise RuntimeError(label+' rc='+str(p.returncode))
try:
 if not probe_asset.is_file() or probe_asset.stat().st_size!=5656 or hashlib.sha256(probe_asset.read_bytes()).hexdigest()!=probe_sha:raise RuntimeError('QNN probe asset missing or mismatched: '+str(probe_asset))
 state('NATIVE_BUILD')
 native=app/'native';run('NATIVE_BUILD',['cmd.exe','/d','/c',tool/'android-sdk/ndk/27.3.13750724/ndk-build.cmd','NDK_PROJECT_PATH='+str(native),'APP_BUILD_SCRIPT='+str(native/'jni/Android.mk'),'NDK_APPLICATION_MK='+str(native/'jni/Application.mk'),'NDK_OUT='+str(out/'native-obj'),'NDK_LIBS_OUT='+str(out/'native-libs'),'-j2'],native)
 so=out/'native-libs/arm64-v8a/librinqnnbridge.so';assert so.is_file();shutil.copy2(so,app/'src/main/jniLibs/arm64-v8a/librinqnnbridge.so')
 run('KOTLIN_TEST_AND_APK',[tool/'gradle-8.13/bin/gradle.bat','testDebugUnitTest','assembleDebug','--no-daemon','--max-workers=2','--console=plain','--stacktrace'])
 apk=out/f'Rin-NPU-Agent-v{version}-debug-runtime-ready.apk'
 candidates=list((app.parents[1]/'toolchain/temp/integration-1.6-gradle-build/outputs/apk/debug').glob('*.apk'));assert len(candidates)==1,candidates
 shutil.copy2(candidates[0],apk)
 state('DSP_AND_SIGNATURE')
 sys.path.insert(0,str(tool));import postprocess_dsp_apk as post
 post.TOOL=out/'postprocess'
 post.run_checked=lambda cmd:run('DSP_AND_SIGNATURE',cmd)
 post.patch_apk(apk)
 report={'version':version,'code':code,'application_id':'com.geniex.demo','apk':str(apk),'bytes':apk.stat().st_size,'sha256':hashlib.sha256(apk.read_bytes()).hexdigest(),'phone_tested':False}
 with zipfile.ZipFile(apk) as z:
  probe=z.read('assets/sdxl_runtime/qnn_probe/libCalculator_skel.so');assert len(probe)==5656 and hashlib.sha256(probe).hexdigest()==probe_sha
  report['qnn_probe_asset_sha256']=probe_sha
  assert b'Java_com_geniex_demo_image_QnnInProcessNative_runLoraSequence' in z.read('lib/arm64-v8a/librinqnnbridge.so')
  j=json.loads(z.read('assets/lora_selftest/manifest.json'))
  for f in j['assets']:assert hashlib.sha256(z.read('assets/lora_selftest/'+f['name'])).hexdigest()==f['sha256'],f['name']
  report['selftest_assets']=len(j['assets'])
  dynamic=json.loads(z.read('assets/lora_dynamic_selftest/manifest.json'))
  for f in dynamic['assets']:assert hashlib.sha256(z.read('assets/lora_dynamic_selftest/'+f['name'])).hexdigest()==f['sha256']
  report['dynamic_assets']=len(dynamic['assets'])
 (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');state('COMPLETE');print(report,flush=True)
except Exception as e:
 state('FAILED '+str(e));traceback.print_exc();sys.exit(1)
