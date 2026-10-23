#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize paired host recording runs; these are not CTS GPU results."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def arm(directory, expected):
    cases = {}
    submissions = 0
    inputs = {}
    for path in sorted(directory.glob('batch-*/SUMMARY.json')):
        summary = json.loads(path.read_text())
        assert not summary['gpu_execution'] and not summary['queue_submission']
        assert not summary['timed_out'] and summary['returncode'] in (0, 1)
        assert not summary['api_errors'] and not summary['debug_errors']
        manifest = json.loads((path.parent / 'RUN.json').read_text())
        for key in ('driver', 'proxy', 'shim', 'cts'):
            value = manifest['inputs'][key]
            assert inputs.setdefault(key, value) == value
        submissions += summary['suppressed_submissions']
        for row in summary['cases']:
            assert row['case'] not in cases, row['case']
            cases[row['case']] = row
    assert set(cases) == expected, (len(cases), len(expected))
    return cases, dict(cases=len(cases), api_errors=0, debug_errors=0,
                      suppressed_submissions=submissions, inputs=inputs,
                      harness_outcomes=dict(Counter(r['status'] for r in cases.values())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--emulated', type=Path, required=True)
    a = p.parse_args()
    rows = list(csv.DictReader(a.inventory.read_text().splitlines(), delimiter='\t'))
    expected = {row['case'] for row in rows}
    assert len(expected) == len(rows)
    off, off_summary = arm(a.emulated, expected)
    on, on_summary = arm(a.native, expected)
    assert off_summary['inputs'] == on_summary['inputs']
    changed = [name for name in sorted(expected) if on[name] != off[name]]
    deferred = [row['case'] for row in rows if row['api'] == 'endCommandBuffer'
                and row['first_bucket'] in ('points', 'triangle_output_shape')]
    print(json.dumps(dict(
        gpu_execution=False, queue_submission=False,
        comparison='same hybrid candidate, native disabled versus enabled',
        outcome_note='Harness statuses include expected missing-GPU image failures; '
                     'they are not CTS device pass/fail results.',
        inventory_sha256=hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
        native_off=off_summary, native_on=on_summary,
        former_deferred_shape_cases_recorded=len(deferred),
        changed_outcomes=changed), indent=2))
    if changed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
