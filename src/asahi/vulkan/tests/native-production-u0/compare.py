#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare our compiler streams: only U0 retirement and its branch relocation."""
import argparse
import difflib
import json
from pathlib import Path
import re


def programs(path, names):
    result = []
    rows = None
    for line in Path(path).read_text().splitlines():
        if line.startswith('APPLE9_ASM_BEGIN'):
            assert rows is None
            rows = []
        elif line == 'APPLE9_ASM_END':
            result.append(rows)
            rows = None
        elif line.startswith('APPLE9_ASM '):
            row = dict(x.split('=', 1) for x in line.split()[1:])
            row['op'] = names[int(row['op'])]
            row['offset'] = int(row['offset'])
            row['code'] = bytes.fromhex(row.pop('bytes'))
            rows.append(row)
    assert rows is None
    return result


def compare(before, after):
    def key(r):
        return json.dumps({k: v for k, v in r.items() if k not in ['offset', 'code', 'bytes']}, sort_keys=True)
    matcher = difflib.SequenceMatcher(a=list(map(key, before)), b=list(map(key, after)), autojunk=False)
    pairs, added = [], []
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == 'equal':
            pairs.extend(zip(range(a, b), range(c, d)))
        else:
            assert tag == 'insert', (tag, a, b, c, d)
            for j in range(c, d):
                r = after[j]
                assert r['op'] == 'WAIT' and r['code'] == bytes.fromhex('060c'), r
                assert str(r['imm']) in ['3', '0x3'] and r['wait'] == r['producer'] == '0'
                added.append(dict(offset=r['offset'], mask=3, before=after[j+1]['op'] if j+1 < len(after) else 'END'))
    relocations = []
    mapping = {before[a]['offset']: after[b]['offset'] for a, b in pairs}
    if before:
        mapping[before[-1]['offset'] + len(before[-1]['code'])] = after[-1]['offset'] + len(after[-1]['code'])
    # A branch to the consumer's block starts at its newly inserted WAIT.
    for new in added:
        end = new['offset'] + 2
        for old, target in list(mapping.items()):
            if target == end:
                mapping[old] = new['offset']
    for a, b in pairs:
        old, new = before[a], after[b]
        if old['op'] in ['JMP_EXEC_ANY', 'JMP_EXEC_NONE']:
            assert len(old['code']) == len(new['code']) == 10
            assert old['code'][:3] == new['code'][:3] and old['code'][9:] == new['code'][9:]
            target = old['offset'] + int.from_bytes(old['code'][3:9], 'little', signed=True)
            new_target = new['offset'] + int.from_bytes(new['code'][3:9], 'little', signed=True)
            assert mapping[target] == new_target, (old, new, target, new_target)
            if old['code'] != new['code']:
                relocations.append(dict(old_offset=old['offset'], new_offset=new['offset'], old_target=target, new_target=new_target))
        else:
            assert old['code'] == new['code'], (old, new)
    assert len(after) == len(before) + len(added)
    return dict(added_retirements=added, branch_relocations=relocations,
                other_instruction_changes=0, bytes_added=2*len(added))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', required=True, type=Path)
    p.add_argument('--after', required=True, type=Path)
    p.add_argument('--fixtures', required=True, type=Path)
    p.add_argument('--header', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    body = a.header.read_text().split('enum agx_apple9_vir_opcode {')[1].split('};')[0]
    names = [x.strip().removeprefix('AGX_APPLE9_VIR_') for x in re.sub(r'/\*.*?\*/', '', body, flags=re.S).split(',') if x.strip()]
    before, after = programs(a.before, names), programs(a.after, names)
    fixtures = [x for x in a.fixtures.read_text().splitlines() if x.startswith('fixture=')]
    assert len(before) == len(after) == len(fixtures)
    rows = [dict(fixture=f, **compare(old, new)) for f, old, new in zip(fixtures, before, after)]
    result = dict(programs=len(rows), changed=sum(bool(x['added_retirements']) for x in rows),
                  only_new_retirement_and_branch_relocation=True, fixtures=rows)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(result['programs'], 'programs;', result['changed'], 'changed; only new WAIT 0x03 and required branch relocations')

if __name__ == '__main__':
    main()
