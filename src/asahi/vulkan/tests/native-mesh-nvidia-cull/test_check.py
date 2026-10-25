#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Host-only checker controls; fabricated audit lines are never device evidence."""
import tempfile
import unittest
from pathlib import Path
import check
import run


class Checker(unittest.TestCase):
    def test_oracle_and_rejections(self):
        self.assertEqual(check.mutations()['retained'], dict(visible=126,mixed=84,culled=0))
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            for profile in check.PROFILES:
                image=check.expected(profile);buffer=check.inputs(profile)
                (d/'pixels.bin').write_bytes(image);(d/'Q.bin').write_bytes(buffer)
                self.assertEqual(check.verify(d,profile)['verdict'],'PASS')
                for offset in (0,1032,len(image)-1):
                    bad=bytearray(image);bad[offset]^=1
                    (d/'pixels.bin').write_bytes(bad)
                    self.assertEqual(check.verify(d,profile)['verdict'],'FAIL')
                (d/'pixels.bin').write_bytes(image)
                for offset in (4096,8192,12288+9*16,16384+224,len(buffer)-1):
                    bad=bytearray(buffer);bad[offset]^=1;(d/'Q.bin').write_bytes(bad)
                    self.assertEqual(check.verify(d,profile)['verdict'],'FAIL')

    def test_audit_and_rejections(self):
        lib=Path('/kit/lib/libvulkan_asahi.so');rev='0123456789abcdef'
        prefix='renderer=Apple M4 Honeykrisp git-'+rev[:10]+'\n'
        for phase in ('before','after'):
            prefix+='VM_AUDIT '+phase+' fd=7 dev=226:128 result=0 asahi-native-mesh v1 generation=5 profile=research classes=-\n'
        prefix+='VM_MAP '+str(lib)+'\n'
        zero=dict.fromkeys(run.COUNTERS,0);one=dict.fromkeys(run.COUNTERS,1)
        for arm,route,after in [('native','1 T=32',one),('emulated','0 T=0',zero)]:
            log=prefix+'HK_MESH_ROUTE native='+route+' V=64 P=126\n'
            audit=lambda text,end:run.audit(text,'task1-mixed',arm,rev,zero,end,lib)
            self.assertEqual(audit(log,after)['verdict'],'PASS')
            for bad in (log.replace('VM_AUDIT','MISSING'),log.replace('profile=research','profile=none'),
                        log.replace('generation=5','generation=72057594037927936'),
                        log.replace('V=64','V=32'),log.replace(str(lib),'/wrong'),
                        log.replace('Honeykrisp','wrong'),log+log):
                self.assertEqual(audit(bad,after)['verdict'],'FAIL')
            self.assertEqual(audit(log,zero if arm=='native' else one)['verdict'],'FAIL')

    def test_arm_selection(self):
        for arm in run.ARMS:
            env=run.environment(Path('/out'),Path('/out')/arm,arm)
            enabled='1' if arm=='native' else '0'
            for key in ('HK_APPLE9_NATIVE_INDEX','HK_APPLE9_NATIVE_MESH','HK_APPLE9_NATIVE_TASK_BRIDGE'):
                self.assertEqual(env[key],enabled)
            self.assertEqual(env['MESA_SHADER_CACHE_DISABLE'],'true')


if __name__=='__main__':unittest.main()
