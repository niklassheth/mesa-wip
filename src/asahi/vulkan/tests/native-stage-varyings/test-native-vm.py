#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the admission probe without opening a device."""
import ctypes
import errno
import importlib.util
from pathlib import Path
import stat
import struct
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("native_vm", Path(__file__).with_name("native-vm.py"))
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)
PROFILE = "asahi-native-mesh v1 generation=17 profile=research classes=-"


class Lifecycle(unittest.TestCase):
    def run_probe(self, fail=None, after=PROFILE, generation=16, size=0x20000000,
                  start=0x4000, end=0x10000000000, mode=stat.S_IFCHR):
        calls = []

        def ioctl(fd, request, arg):
            self.assertEqual(fd, 10)
            calls.append(request)
            if request == fail:
                raise OSError(errno.ENOTTY, "unsupported ioctl")
            if request == vm.GET_PARAMS:
                group, pad, pointer, length = struct.unpack("<IIQQ", arg)
                self.assertEqual((group, pad, length), (0, 0, ctypes.sizeof(vm.ParamsPrefix)))
                params = vm.ParamsPrefix.from_address(pointer)
                params.gpu_generation = generation
                params.gpu_variant = ord("G")
                params.vm_start, params.vm_end = start, end
                params.vm_kernel_min_size = size
            elif request == vm.VM_CREATE:
                self.assertEqual(struct.unpack("<QQII", arg), (end - size, end, 0, 0))
                struct.pack_into("<I", arg, 16, 3)
            elif request == vm.VM_DESTROY:
                self.assertEqual(struct.unpack("<II", arg), (3, 0))
            else:
                self.fail(f"unexpected ioctl {request:x}")
            return 0

        st = types.SimpleNamespace(st_mode=mode, st_rdev=vm.os.makedev(226, 128))
        with patch.object(vm.os, "open", return_value=10), \
             patch.object(vm.os, "close") as close, \
             patch.object(vm.os, "fstat", return_value=st), \
             patch.object(vm.Path, "read_text", side_effect=[PROFILE, after]), \
             patch.object(vm.fcntl, "ioctl", side_effect=ioctl):
            result = vm.probe()
            close.assert_called_once_with(10)
        self.assertEqual(result["gpu_submissions"], 0)
        return result, calls

    def test_lifecycle(self):
        result, calls = self.run_probe()
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(calls, [vm.GET_PARAMS, vm.VM_CREATE, vm.VM_DESTROY])
        self.assertTrue(result["vm_created"] and result["vm_destroyed"])

    def test_unavailable_operations_never_pass(self):
        for fail in (vm.GET_PARAMS, vm.VM_CREATE, vm.VM_DESTROY):
            with self.subTest(fail=fail):
                result, calls = self.run_probe(fail=fail)
                self.assertEqual(result["verdict"], "FAIL")
                self.assertEqual(result["errors"][0]["errno"], errno.ENOTTY)
                self.assertEqual(calls[-1], fail)
                self.assertEqual(result["vm_created"], fail == vm.VM_DESTROY)
                self.assertFalse(result["vm_destroyed"])

    def test_generation_change_still_destroys_vm(self):
        result, calls = self.run_probe(after=PROFILE.replace("=17 ", "=18 "))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["errors"][0]["operation"], "profile-after")
        self.assertEqual(calls[-1], vm.VM_DESTROY)
        self.assertTrue(result["vm_destroyed"])

    def test_non_render_node(self):
        result, calls = self.run_probe(mode=stat.S_IFREG)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(calls, [])

    def test_bad_parameters_do_not_create_vm(self):
        for kwargs in ({"generation": 13}, {"size": 0}, {"size": 1},
                       {"size": 0x20000000000}, {"start": 0x10000000000},
                       {"end": 0x10000000001}):
            with self.subTest(kwargs=kwargs):
                result, calls = self.run_probe(**kwargs)
                self.assertEqual(result["verdict"], "FAIL")
                self.assertEqual(calls, [vm.GET_PARAMS])


if __name__ == "__main__":
    unittest.main()
