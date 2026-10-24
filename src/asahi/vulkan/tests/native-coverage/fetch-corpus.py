#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch pinned public corpus bytes, verifying SHA-256 and Git blob identity."""
import argparse, concurrent.futures, hashlib, json
from pathlib import Path
import urllib.request
p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('SOURCE-MANIFEST.json'))
a=p.parse_args();rows=json.loads(a.manifest.read_text())
def one(row):
 dest=a.out/'sources'/row['repository'].replace('/','-')/row['path']
 data=dest.read_bytes() if dest.exists() else urllib.request.urlopen(row['url'],timeout=60).read()
 assert hashlib.sha256(data).hexdigest()==row['sha256'],row['url']
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob'],row['url']
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(one,rows))
a.out.mkdir(exist_ok=True,parents=True);(a.out/'SOURCE-MANIFEST.json').write_text(a.manifest.read_text())
print('verified',len(rows),'public source/binary/license files')
