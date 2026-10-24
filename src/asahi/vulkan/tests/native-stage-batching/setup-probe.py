#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel-independent setup checks; never opens a GPU or writes a knob."""
import os
from pathlib import Path
import shutil
import sys
import subprocess

here = Path(__file__).resolve().parent
try:
    assert os.uname().machine == 'aarch64'
    assert os.sysconf('SC_PAGESIZE') == 16384
    assert not (Path.home()/'.drirc').exists(), 'unexpected per-user drirc'
    assert shutil.disk_usage(here).free >= 1024*1024*1024
    for name in ('bin/gs-transitions','bin/tes-transitions'):
        assert os.access(here/name, os.X_OK), name
    # Resolve dependencies with the target loader; no Vulkan initialization.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(
        ('LD_', 'VK_', 'HK_', 'MESA_', 'AGX_', 'ASAHI_', 'DRI'))}
    for name in ('bin/gs-transitions', 'bin/tes-transitions',
                 'lib/vm-audit.so', 'lib/libvulkan_asahi.so'):
        result = subprocess.run(['/lib/ld-linux-aarch64.so.1', '--list', str(here/name)],
                                env=env, text=True, capture_output=True, timeout=10)
        assert result.returncode == 0 and 'not found' not in result.stdout+result.stderr, (
            name, result.returncode, result.stdout, result.stderr)
    print('SETUP_PASS: architecture, dependencies, free space and per-user drirc')
except Exception as error:
    print('SETUP_FAIL: '+str(error))
    sys.exit(1)
