#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic parser negatives, not device evidence."""
from pathlib import Path
import sys
import unittest
sys.path.append(str(Path(__file__).resolve().parent.parent/'native-stage-batching'))
import run

class Audit(unittest.TestCase):
    def setUp(self):
        self.case = run.cases()[0]
        self.before = dict(native_renders=10, native_dispatches=20)
        self.after = dict(native_renders=11, native_dispatches=21)
        self.log = 'renderer=Apple M4 Honeykrisp git-0123456789\n'
        for fd in (7, 8):
            for stage in ('before', 'after'):
                self.log += f'VM_AUDIT {stage} fd={fd} dev=226:128 result=0 asahi-native-mesh v1 generation=3 profile=qualified classes={run.CLASSES}\n'
        self.log += 'VM_MAP ' + str(run.HERE/'lib/libvulkan_asahi.so') + '\n'
        self.log += 'HK_MESH_ROUTE native=1 T=16 V=3 P=1\n'

    def check(self, log=None, after=None):
        return run.audit(self.log if log is None else log, self.case, '0123456789abc',
                         self.before, self.after if after is None else after)

    def test_valid(self):
        self.assertEqual(self.check()['counters'], dict(native_renders=1, native_dispatches=1))

    def test_missing_audit(self):
        with self.assertRaisesRegex(RuntimeError, 'missing own-FD'):
            self.check('\n'.join(s for s in self.log.splitlines() if not s.startswith('VM_AUDIT')))

    def test_profile_generation(self):
        for old, new in [('generation=3', 'generation=4'), ('profile=qualified', 'profile=research')]:
            with self.subTest(new=new), self.assertRaises(RuntimeError):
                self.check(self.log.replace(old, new, 1))

    def test_wrong_map(self):
        with self.assertRaisesRegex(RuntimeError, 'mapped driver'):
            self.check(self.log.replace(str(run.HERE/'lib/libvulkan_asahi.so'), '/usr/lib64/libvulkan_asahi.so'))

    def test_wrong_route(self):
        for replacement in ('native=0 T=16 V=3 P=1', 'native=1 T=32 V=3 P=1'):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(RuntimeError, 'route'):
                self.check(self.log.replace('native=1 T=16 V=3 P=1', replacement))

    def test_admission_counts(self):
        for after in (self.before, dict(native_renders=11,native_dispatches=22)):
            with self.subTest(after=after), self.assertRaisesRegex(RuntimeError, 'counter mismatch'):
                self.check(after=after)

    def test_counter_whitespace_and_duplicates(self):
        self.assertEqual(run.counts('native_renders 2\nnative_dispatches\t3\n'),
                         dict(native_renders=2,native_dispatches=3))
        with self.assertRaises(RuntimeError):
            run.counts('native_renders 2\nnative_renders 2\nnative_dispatches 3\n')

    def test_profiles_and_default_environment(self):
        env=run.environment(Path('/tmp/m2'),self.case)
        self.assertNotIn('HK_APPLE9_NATIVE_MESH',env)
        self.assertNotIn('HK_APPLE9_NATIVE_MESH_PACKED_SHAPES',env)
        self.case=run.cases()[-1]
        self.log=self.log.replace('profile=qualified classes='+run.CLASSES,'profile=research classes=-').replace('T=16','T=32')
        self.check()
        self.assertEqual(run.environment(Path('/tmp/m2'),self.case)['HK_APPLE9_NATIVE_MESH'],'1')

    def test_renderer_pin(self):
        with self.assertRaisesRegex(RuntimeError, 'renderer/revision'):
            self.check(self.log.replace('0123456789', 'abcde'))

if __name__ == '__main__':
    unittest.main()
