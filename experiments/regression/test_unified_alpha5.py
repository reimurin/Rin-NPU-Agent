from pathlib import Path
import argparse,json,shutil,sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/main/assets/sdxl_runtime'),str(ROOT/'model_tools/partitioned'),str(ROOT/'experiments/regression')]
import test_unified_alpha4 as alpha4
import rin_unified as unified

class Alpha5Tests(alpha4.UnifiedTests):
    def clone_resolution(self,width,height):
        target=self.base/unified.relative_for(width,height)
        shutil.copytree(self.template,target)
        mp=target/'lora_template.json';data=json.loads(mp.read_text())
        data['model_id']=unified.model_id_for(width,height);data['resolution']=[width,height]
        mp.write_text(json.dumps(data),encoding='utf-8')
        return target


    def shared_pack(self):
        root=self.base/'context/model_packs';root.mkdir(parents=True,exist_ok=True)
        folder=root/'shared-v1';folder.mkdir()
        # Reuse the actual synthetic partition contexts from the inherited fixture.
        files={}
        for pid in ('encoder_p0','encoder_p1','decoder_p0'):
            src=self.template/(pid+'.bin');dst=folder/(pid+'.bin');shutil.copy2(src,dst)
            files[pid]={'file':dst.name,'bytes':dst.stat().st_size,'sha256':alpha4.baseline.runtime.digest(dst)}
        # shared_runtime intentionally requires the production seven-part inventory.
        for pid in ('encoder_p2','decoder_p1','decoder_p2','decoder_p3'):
            dst=folder/(pid+'.bin');dst.write_bytes(('dummy-'+pid).encode());files[pid]={'file':dst.name,'bytes':dst.stat().st_size,'sha256':alpha4.baseline.runtime.digest(dst)}
        vae=folder/'vae.bin';vae.write_bytes(b'fixture-vae')
        resolutions=[]
        for width,height in ((1024,1024),(832,1216),(1216,832)):
            data=json.loads(self.mp.read_text())
            data['model_id']=unified.model_id_for(width,height);data['resolution']=[width,height]
            for stage in ('encoder','decoder'):
                for spec in data['graphs'][stage]['parts']:
                    item=files[spec['id']];spec['context_file']=item['file'];spec['context_bytes']=item['bytes'];spec['context_sha256']=item['sha256']
            name=f'template_{width}x{height}.json';(folder/name).write_text(json.dumps(data),encoding='utf-8')
            resolutions.append({'width':width,'height':height,'graph':f'_{width}x{height}','template':name})
        manifest={'schema':1,'complete':True,'runtime_abi':1,'id':'shared-v1','version':'1.0.0','contexts_root':'.',
                  'target':{'qnn_soc_id':69,'dsp_arch':79},'lora':{'external_dynamic_ab':True,'rank_capacity':64,'abi_signature':'synthetic-abi'},
                  'contexts':files,'vae':{'file':'vae.bin','bytes':vae.stat().st_size},'resolutions':resolutions}
        (folder/'model_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        (root/'current.json').write_text(json.dumps({'directory':folder.name}),encoding='utf-8')
        return folder

    def test_30_resolution_specific_component_selected(self):
        target=self.clone_resolution(832,1216)
        unified.set_active(self.base,True,'test',(),832,1216)
        plan=unified.prepare('cat','',self.base,832,1216,log=lambda _:None)
        self.assertTrue(plan.metadata['unified_model'])
        self.assertEqual(plan.metadata['resolution'],[832,1216])
        self.assertEqual(plan.metadata['model_id'],'wai-v170-sm8750-lora-r64-832x1216-partitioned-v1')
        self.assertTrue(all(str(target) in path for path in plan.part_contexts.values()))


    def test_33_alpha6_shared_pack_selects_all_native_graphs(self):
        folder=self.shared_pack()
        for width,height in ((1024,1024),(832,1216),(1216,832)):
            info=unified.shared_runtime(self.base,width,height)
            self.assertIsNotNone(info);self.assertEqual(info['graph'],f'_{width}x{height}')
            self.assertEqual(Path(info['context_root']).resolve(),folder.resolve())
            plan=unified.prepare('cat','',self.base,width,height,log=lambda _:None)
            self.assertTrue(plan.metadata['shared_model_pack']);self.assertEqual(plan.metadata['graph_name'],f'_{width}x{height}')
            self.assertFalse(unified.activation_path(self.base,width,height).exists())
            self.assertTrue(all(folder.resolve() in Path(path).resolve().parents for path in plan.part_contexts.values()))

    def test_34_alpha6_shared_partition_execution_forwards_graph_name(self):
        self.shared_pack();plan=unified.prepare('cat','',self.base,832,1216,log=lambda _:None)
        self.run_plan(plan,rows=1)
        self.assertTrue(self.calls)
        self.assertTrue(all(kwargs.get('graph_name')=='_832x1216' for _,kwargs in self.calls))

    def test_36_alpha6_retained_pack_without_current_is_not_activated(self):
        folder=self.shared_pack();(folder.parent/'current.json').unlink()
        self.assertEqual(unified.shared_resolutions(self.base),[])
        self.assertIsNone(unified.shared_runtime(self.base,1024,1024))

    def test_35_alpha6_current_pointer_excludes_stale_pack(self):
        current=self.shared_pack();root=current.parent
        stale=root/'stale';shutil.copytree(current,stale)
        data=json.loads((stale/'model_manifest.json').read_text());data['resolutions']=[{'width':768,'height':1344,'graph':'_768x1344','template':'template_1024x1024.json'}]
        (stale/'model_manifest.json').write_text(json.dumps(data),encoding='utf-8')
        self.assertEqual(set(unified.shared_resolutions(self.base)),{(1024,1024),(832,1216),(1216,832)})

    def test_31_alpha4_memory_rollback_migrates_to_safe_unified(self):
        unified.set_active(self.base,False,'alpha4_persistent_memory_rollback_6ctx_519MiB',['encoder_p0'])
        plan=unified.prepare('cat','',self.base,1024,1024,log=lambda _:None)
        config=unified.activation_config(self.base)
        self.assertTrue(config['active'])
        self.assertEqual(config['persistent_parts'],[])
        self.assertEqual(config['reason'],'alpha5_migrated_safe_nonpersistent')
        self.assertTrue(plan.metadata['unified_model'])
        self.assertEqual(plan.metadata['persistent_parts'],[])

    def test_32_unrelated_disabled_reason_stays_disabled(self):
        unified.set_active(self.base,False,'runtime_validation_failed')
        plan=unified.prepare('cat','',self.base,1024,1024,log=lambda _:None)
        self.assertFalse(unified.active(self.base))
        self.assertFalse(plan.metadata.get('unified_model',False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();OUT=Path(args.out)
    if OUT.exists():raise SystemExit('Choose new test directory')
    OUT.mkdir(parents=True);alpha4.baseline.OUT=OUT/'cases';alpha4.baseline.OUT.mkdir()
    with (OUT/'tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Alpha5Tests))
    report={'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'scope':'alpha5 resolution-aware unified routing plus inherited alpha4/alpha3 synthetic partition CPU numerical tests; not handset NPU'}
    (OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print((OUT/'tests.log').read_text(encoding='utf-8'));print(report);raise SystemExit(not result.wasSuccessful())
