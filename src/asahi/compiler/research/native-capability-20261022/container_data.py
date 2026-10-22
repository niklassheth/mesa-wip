#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Parse own-MSL Mach-O containers and compiler metadata; no instruction decoding."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

def digest(data):
    return hashlib.sha256(data).hexdigest()

def metadata(data):
    assert data[:4] == bytes.fromhex("cffaedfe") and len(data) < 64 << 20
    count, extent = struct.unpack_from("<II", data, 16)
    assert count < 256 and 32 + extent <= len(data)
    at, sections, symbols, symtab = 32, [], [], None
    for _ in range(count):
        command, size = struct.unpack_from("<II", data, at)
        assert size >= 8 and at + size <= 32 + extent
        if command == 0x19:
            n = struct.unpack_from("<I", data, at + 64)[0]
            assert n < 64 and 72 + 80 * n <= size
            for i in range(n):
                p = at + 72 + 80 * i
                name = data[p:p+16].rstrip(b"\0").decode()
                segment = data[p+16:p+32].rstrip(b"\0").decode()
                address, length, offset = struct.unpack_from("<QQI", data, p+32)
                relocations = struct.unpack_from("<I", data, p+60)[0]
                assert offset + length <= len(data)
                sections.append(dict(name=name, segment=segment, address=address,
                                     length=length, offset=offset, relocations=relocations))
        elif command == 2:
            symtab = struct.unpack_from("<4I", data, at+8)
        at += size
    assert at == 32 + extent
    if symtab:
        offset, n, strings, length = symtab
        assert n < 10000 and offset + 16*n <= len(data) and strings + length <= len(data)
        for i in range(n):
            index, kind, section, desc, value = struct.unpack_from("<IBBHQ", data, offset+16*i)
            assert index < length
            end = data.index(b"\0", strings+index, strings+length)
            name = data[strings+index:end].decode()
            if name in ("_agc.main", "_agc.main.constant_program"):
                assert all(s["name"] != name for s in symbols)
                symbols.append(dict(name=name, section=section, value=value))
    return sections, symbols

def native(raw):
    if raw[:4] not in (bytes.fromhex("cafebabe"), bytes.fromhex("cbfebabe")):
        return raw
    count, = struct.unpack_from(">I", raw, 4)
    assert count < 16
    found = []
    for i in range(count):
        cpu, sub, offset, size, align = struct.unpack_from(">5I", raw, 8+20*i)
        assert offset + size <= len(raw)
        if cpu == 0x1000013:
            found.append(raw[offset:offset+size])
    assert len(found) == 1
    return found[0]

class Table:
    def __init__(self, data, at=None):
        self.data = data
        self.at = self.u32(0) if at is None else at
        self.vt = self.at - struct.unpack_from("<i", data, self.at)[0]
        assert 0 <= self.vt <= len(data)-4
        self.extent, = struct.unpack_from("<H", data, self.vt)
        assert self.extent >= 4 and self.vt + self.extent <= len(data)

    def u32(self, at):
        assert 0 <= at <= len(self.data)-4
        return struct.unpack_from("<I", self.data, at)[0]

    def offset(self, field):
        if 4 + 2*field >= self.extent:
            return None
        offset, = struct.unpack_from("<H", self.data, self.vt+4+2*field)
        return self.at + offset if offset else None

    def value(self, field, default=0):
        at = self.offset(field)
        return self.u32(at) if at is not None else default

    def table(self, field):
        at = self.offset(field)
        assert at is not None
        return Table(self.data, at+self.u32(at))

    def vector(self, field):
        at = self.offset(field)
        assert at is not None
        at += self.u32(at)
        size = self.u32(at)
        assert at+4+size <= len(self.data)
        return self.data[at+4:at+4+size]

