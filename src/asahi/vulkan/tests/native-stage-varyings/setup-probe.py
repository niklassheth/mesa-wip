#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel-independent setup check; no GPU initialization."""
import os,shutil,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
try:
    assert os.uname().machine=='aarch64' and os.sysconf('SC_PAGESIZE')==16384
    assert not (Path.home()/'.drirc').exists()
    assert shutil.disk_usage(here).free>=1024*1024*1024
    assert os.access(here/'bin/probe',os.X_OK)
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('LD_','VK_','HK_','MESA_','ASAHI_','AGX_','DRI'))}
    for name in ('bin/probe','lib/vm-audit.so'):
        p=subprocess.run(['/lib/ld-linux-aarch64.so.1','--list',str(here/name)],env=env,text=True,capture_output=True,timeout=10)
        assert p.returncode==0 and 'not found' not in p.stdout+p.stderr,(name,p.stdout,p.stderr)
    print('SETUP_PASS')
except Exception as error:
    print('SETUP_FAIL: '+str(error));sys.exit(1)
