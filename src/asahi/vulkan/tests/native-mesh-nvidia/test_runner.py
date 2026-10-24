#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic negative tests of the exact native audit; no GPU evidence."""
from pathlib import Path
import sys
import unittest
sys.path.append(str(Path(__file__).resolve().parent.parent/'native-stage-batching'))
import run

class Audit(unittest.TestCase):
    def setUp(self):
        self.case=run.cases()[0]
        self.before=dict(native_renders=3,native_dispatches=4)
        self.after=dict(native_renders=4,native_dispatches=5)
        self.log='renderer=Apple M4 Honeykrisp git-0123456789\n'
        for fd in (7,8):
            for phase in ('before','after'):
                self.log+=f'VM_AUDIT {phase} fd={fd} dev=226:128 result=0 asahi-native-mesh v1 generation=2 profile=research classes=-\n'
        self.log+='VM_MAP '+str(run.HERE/'lib/libvulkan_asahi.so')+'\n'
        self.log+='HK_MESH_ROUTE native=1 T=32 V=64 P=126\n'

    def audit(self,log=None,after=None):
        return run.audit(self.log if log is None else log,self.case,'0123456789abc',
                         self.before,self.after if after is None else after)

    def test_valid(self):self.assertEqual(self.audit()['counters'],dict(native_renders=1,native_dispatches=1))

    def test_wrong_identity(self):
        for old,new in [('0123456789','fffffffffff'),('profile=research','profile=qualified'),
                        ('generation=2','generation=3'),('classes=-','classes=bad'),
                        (str(run.HERE/'lib/libvulkan_asahi.so'),'/usr/lib64/libvulkan_asahi.so')]:
            with self.subTest(new=new),self.assertRaises(RuntimeError):self.audit(self.log.replace(old,new,1))

    def test_wrong_route(self):
        for old,new in [('native=1','native=0'),('T=32','T=128'),('V=64','V=3'),('P=126','P=1')]:
            with self.subTest(new=new),self.assertRaises(RuntimeError):self.audit(self.log.replace(old,new))

    def test_missing_audit(self):
        with self.assertRaises(RuntimeError):self.audit('\n'.join(l for l in self.log.splitlines() if not l.startswith('VM_AUDIT')))

    def test_counters(self):
        for after in (self.before,dict(native_renders=4,native_dispatches=6)):
            with self.subTest(after=after),self.assertRaises(RuntimeError):self.audit(after=after)

    def test_environment(self):
        env=run.environment(Path('/tmp/native-nvidia'),self.case)
        self.assertEqual(env['HK_APPLE9_NATIVE_MESH'],'1')
        self.assertEqual(env['HK_APPLE9_NATIVE_MESH_ROUNDS'],'32')
        self.assertNotIn('HK_APPLE9_NATIVE_MESH_WIDE',env)

if __name__=='__main__':unittest.main()
