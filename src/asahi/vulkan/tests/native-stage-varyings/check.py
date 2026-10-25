#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent pixel-center barycentric oracle for the authored GS/TES scenes."""
import argparse, json, math, struct
from pathlib import Path
W,H=128,64
CASES=('gs3','gs4','tri','quad')
def expected(case,x,y):
    if not 8 <= y < 56: return (0.,)*4
    if case.startswith('gs'):
        if not 8 <= x < 96: return (0.,)*4
        p=(x-8)//8
        u=((x-8)%8+.5)/8; v=(y-8+.5)/48
        if u+v < 1:
            corners=(0,1,2); weights=(1-u-v,u,v); provoking=0
        elif case=='gs4':
            corners=(2,1,3); weights=(1-u,1-v,u+v-1); provoking=1
        else: return (0.,)*4
        xy=((0,0),(1,0),(0,1),(1,1)); ws=(1,2,.5,1.5)
        perspective=[b/ws[c] for b,c in zip(weights,corners)]
        denominator=sum(perspective)
        red=sum(b*(.125+p/32+.25*xy[c][0]) for b,c in zip(perspective,corners))/denominator
        green=sum(b*(.25+.5*xy[c][1]) for b,c in zip(perspective,corners))/denominator
        return red,green,.125+p/64+.125*u+.25*v,(p*8+provoking+1)/128
    if not 8 <= x < 112: return (0.,)*4
    u=(x-8+.5)/104; v=(y-8+.5)/48
    if case=='tri' and u+v >= 1: return (0.,)*4
    return .1875+.25*u,.28125+.5*v,.125+.125*u+.25*v,7/128

def verify(path,case):
    data=Path(path).read_bytes()
    if len(data)!=W*H*16: raise ValueError('wrong RGBA32F readback size')
    values=struct.unpack('<'+str(W*H*4)+'f',data)
    bad=[]; total=0; max_error=0.
    for y in range(H):
        for x in range(W):
            for c,want in enumerate(expected(case,x,y)):
                got=values[(y*W+x)*4+c]
                error=abs(got-want)
                if math.isfinite(got): max_error=max(max_error,error)
                if not math.isfinite(got) or error > (2e-5 if want else 0):
                    total+=1
                    if len(bad)<8: bad.append(dict(x=x,y=y,channel=c,actual=got,expected=want))
    return dict(verdict='FAIL' if total else 'PASS',components=len(values),mismatches=total,max_error=max_error,examples=bad)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=CASES);p.add_argument('pixels',type=Path)
    a=p.parse_args();result=verify(a.pixels,a.case);print(json.dumps(result,indent=2));raise SystemExit(result['verdict']!='PASS')
