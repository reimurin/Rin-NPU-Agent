"""Actual driver resolution functions, synthetic filenames only; never invokes QNN."""
from pathlib import Path
import ast,os,json,argparse,tempfile
root=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
source=(root/'src/main/assets/sdxl_runtime/phone_generate.py').read_text(encoding='utf-8');tree=ast.parse(source)
names={'_discover_available_resolutions','_resolve_contexts','generate'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==3
checks=[]
with tempfile.TemporaryDirectory(prefix='resolution_fixture_',dir=out) as tmp:
 ns={'os':os,'DR':tmp,'SDXL_QNN_LORA_SLOT':''};exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_driver','exec'),ns)
 ctx=Path(tmp)/'context';ctx.mkdir();names=['unet_encoder_fp16.serialized.bin.bin','unet_decoder_fp16.serialized.bin.bin','vae_decoder.serialized.bin.bin']
 def check(name,result):assert result,name;checks.append(name)
 def complete(d):
  d.mkdir(exist_ok=True)
  for name in names:(d/name).write_bytes(b'x')
 check('no_files_no_capabilities',ns['_discover_available_resolutions']()==[])
 complete(ctx);check('legacy_flat_1024_only',ns['_discover_available_resolutions']()==[(1024,1024)])
 paths=ns['_resolve_contexts'](1216,832);check('unavailable_rect_does_not_use_flat',all('/1216x832/' in paths[k].replace('\\','/') for k in ['encoder','decoder','vae']))
 try:ns['generate']('test',width=1216,height=832)
 except ValueError as e:check('unsupported_rejected_before_inference','not installed' in str(e))
 else:raise AssertionError('Unsupported size accepted')
 complete(ctx/'832x1216');check('native_bucket_discovered',(832,1216) in ns['_discover_available_resolutions']())
 (ctx/'832x1216'/names[0]).write_bytes(b'');check('zero_byte_bucket_excluded',(832,1216) not in ns['_discover_available_resolutions']())
 complete(ctx/'99999999999999999999x1024');check('oversized_dimensions_excluded',ns['_discover_available_resolutions']()==[(1024,1024)])
 (ctx/'1024x1024').mkdir();paths=ns['_resolve_contexts'](1024,1024);check('partial_duplicate_bucket_keeps_flat',all('/1024x1024/' not in paths[k].replace('\\','/') for k in ['encoder','decoder','vae']))
 complete(ctx/'1024x1024');check('deduplicate_1024',ns['_discover_available_resolutions']()==[(1024,1024)])
 paths=ns['_resolve_contexts'](1024,1024);check('complete_bucket_preferred',all('/1024x1024/' in paths[k].replace('\\','/') for k in ['encoder','decoder','vae']))
 generate=next(n for n in nodes if n.name=='generate')
 check('no_generation_resize',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='resize' for n in ast.walk(generate)))
 check('no_generation_snap',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_snap_to_nearest_resolution' for n in ast.walk(generate)))
report={'scope':'actual driver, synthetic file fixtures, no model execution','passed':len(checks),'checks':checks};(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)
