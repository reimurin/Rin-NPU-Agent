from pathlib import Path
import argparse,json,sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/main/assets/sdxl_runtime'),str(ROOT/'model_tools/partitioned'),str(ROOT/'experiments/regression')]
import test_partitioned_lora as baseline
import rin_unified as unified

class UnifiedTests(baseline.Tests):
    def test_25_active_unified_no_tags_uses_all_stages(self):
        unified.set_active(self.base,True,'test')
        plan=unified.prepare('cat','',self.base,1024,1024,log=lambda _:None)
        self.assertTrue(plan.metadata['unified_model'])
        self.assertTrue(plan.metadata['zero_delta'])
        self.assertFalse(plan.metadata['active'])
        self.assertEqual(set(plan.contexts),{'encoder','decoder'})
        self.assertEqual(set(plan.part_contexts),{'encoder_p0','encoder_p1','decoder_p0'})

    def test_26_zero_weight_is_same_unified_zero_delta_path(self):
        unified.set_active(self.base,True,'test')
        plan=unified.prepare('cat <lora:absent:0>','',self.base,1024,1024,log=lambda _:None)
        self.assertTrue(plan.metadata['unified_model'])
        self.assertTrue(plan.metadata['zero_delta'])
        self.assertEqual(set(plan.contexts),{'encoder','decoder'})

    def test_27_corrupt_component_auto_rolls_back_normal_generation(self):
        unified.set_active(self.base,True,'test')
        file=self.template/'encoder_p0.bin'
        file.write_bytes(b'x'*file.stat().st_size)
        plan=unified.prepare('cat','',self.base,1024,1024,log=lambda _:None)
        self.assertTrue(plan.metadata['rollback'])
        self.assertFalse(unified.active(self.base))
        self.assertEqual(plan.contexts,{})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();OUT=Path(args.out)
    if OUT.exists():raise SystemExit('Choose new test directory')
    OUT.mkdir(parents=True);baseline.OUT=OUT/'cases';baseline.OUT.mkdir()
    with (OUT/'tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(UnifiedTests))
    report={'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'scope':'alpha4 unified routing plus inherited alpha3 partition loader/executor CPU numerical tests; synthetic contexts, not handset NPU'}
    (OUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print((OUT/'tests.log').read_text(encoding='utf-8'));print(report);raise SystemExit(not result.wasSuccessful())
