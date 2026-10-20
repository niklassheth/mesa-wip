#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""T128 qualification with the previously measured T32/T64 source controls."""
import importlib.util,json
from pathlib import Path

def generate(root):
    path=Path(__file__).resolve().parent.parent/'native-limits/arms.py'
    spec=importlib.util.spec_from_file_location('limits_arms',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=m.generate(root,[
        ('T32_CONTROL',(32,1,1),False,False),
        ('T32_IDS',(32,1,1),True,False),
        ('T64_LOGICAL',(64,1,1),True,False),
        ('T64_LOGICAL_3D',(4,4,4),True,False),
        ('T128_CONTROL',(128,1,1),False,False),
        ('T128_LOGICAL',(128,1,1),True,False),
        ('T127_PARTIAL',(127,1,1),True,False)])
    for row in rows:
        row['diagnostic_only']=False
        row['scalar_pair']={'second_value':0,'explicit_source':True,'physical_relation':'base+1','device_register_measurement':False}
        row['require_preamble']=row['records'] and row['name']!='T127_PARTIAL'
        row['prior_device_result']='Old scalar-single-source kit was confounded; this scalar-pair build is untested'
    (root/'ARMS.json').write_text(json.dumps(dict(no_device_execution=True,research_only=True,arms=rows),indent=2)+'\n')
    (root/'SLOT.json').write_text(json.dumps(dict(
        required_order=['T32_CONTROL','T32_IDS','T64_LOGICAL','T128_CONTROL','T128_LOGICAL','T32_CONTROL'],
        conditional_after_pass=['T64_LOGICAL_3D','T127_PARTIAL'],
        separate_diagnostic=[],repeat_new_outcomes=3,
        policy='Fresh process and paired VS for each draw. Stop required sequence on failure/incomplete. T32_IDS now has an explicit zero neighbor and must pass. Historical failures are not expected failures in this kit.'),indent=2)+'\n')
    return rows
