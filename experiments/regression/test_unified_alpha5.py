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

    def test_30_resolution_specific_component_selected(self):
        target=self.clone_resolution(832,1216)
        unified.set_active(self.base,True,'test',(),832,1216)
        plan=unified.prepare('cat','',self.base,832,1216,log=lambda _:None)
        self.assertTrue(plan.metadata['unified_model'])
        self.assertEqual(plan.metadata['resolution'],[832,1216])
        self.assertEqual(plan.metadata['model_id'],'wai-v170-sm8750-lora-r64-832x1216-partitioned-v1')
        self.assertTrue(all(str(target) in path for path in plan.part_contexts.values()))

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
