#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check that an empty VM can be admitted and reclaimed after a native draw.

This uses the unchanged public DRM UAPI. It submits no GPU commands and never
stands in for the draw's own VM/route audit. In particular, an unsupported
kernel ioctl is a failure on every execution path, including rehearsal.
"""
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys


class ParamsPrefix(ctypes.LittleEndianStructure):
    # drm_asahi_params_global through vm_kernel_min_size. The UAPI permits
    # older/shorter userspace buffers; later fields are not needed here.
    _fields_ = [("features", ctypes.c_uint64)] + [
        (name, ctypes.c_uint32) for name in (
            "gpu_generation", "gpu_variant", "gpu_revision", "chip_id",
            "num_dies", "num_clusters_total", "num_cores_per_cluster",
            "max_frequency_khz")
    ] + [("core_masks", ctypes.c_uint64 * 64)] + [
        (name, ctypes.c_uint64) for name in (
            "vm_start", "vm_end", "vm_kernel_min_size")
    ]


# Current asahi_drm.h: GET_TIME occupies command 1; GET_PARAMS passes its
# output through a pointer and is _IOW, whereas VM_CREATE is _IOWR.
GET_PARAMS = 0x40186440
VM_CREATE = 0xC0186442
VM_DESTROY = 0x40086443


def probe(node="/dev/dri/renderD128"):
    result = dict(verdict="FAIL", gpu_submissions=0, vm_created=False,
                  vm_destroyed=False, errors=[])
    fd = None
    vm_id = None
    operation = "open"

    def require(ok, why):
        if not ok:
            raise ValueError(why)

    try:
        fd = os.open(node, os.O_RDWR | os.O_CLOEXEC)
        st = os.fstat(fd)
        require(stat.S_ISCHR(st.st_mode), "render node is not a character device")
        device = f"{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}"
        profile = Path(f"/sys/dev/char/{device}/device/asahi_native_mesh")
        result["device"] = device

        operation = "get-params"
        params = ParamsPrefix()
        arg = bytearray(struct.pack("<IIQQ", 0, 0, ctypes.addressof(params),
                                    ctypes.sizeof(params)))
        fcntl.ioctl(fd, GET_PARAMS, arg)
        require(params.gpu_generation == 16 and params.gpu_variant == ord("G"),
                "VM lifecycle check requires G16G")
        size = params.vm_kernel_min_size
        end = params.vm_end
        require(size > 0 and size <= end and params.vm_start <= end - size and
                (size | end) & 0x3FFF == 0, "invalid advertised VM range")
        result.update(kernel_start=end - size, kernel_end=end)

        operation = "profile-before"
        before = profile.read_text().strip()
        match = re.fullmatch(
            r"asahi-native-mesh v1 generation=(\d+) profile=research classes=-", before)
        require(match is not None and int(match[1]) < 1 << 56,
                "VM lifecycle check requires the research profile")
        result["before"] = before

        operation = "vm-create"
        create = bytearray(struct.pack("<QQII", end - size, end, 0, 0))
        fcntl.ioctl(fd, VM_CREATE, create)
        vm_id = struct.unpack_from("<I", create, 16)[0]
        result.update(vm_created=True, vm_id=vm_id)
        operation = "profile-after"
        after = profile.read_text().strip()
        result["after"] = after
        require(after == before, "native generation/profile changed at VM creation")
    except (OSError, ValueError) as error:
        result["errors"].append(dict(operation=operation, reason=str(error),
                                     errno=getattr(error, "errno", None)))
    finally:
        if vm_id is not None:
            try:
                fcntl.ioctl(fd, VM_DESTROY, struct.pack("<II", vm_id, 0))
                result["vm_destroyed"] = True
            except OSError as error:
                result["errors"].append(dict(operation="vm-destroy",
                                             reason=str(error), errno=error.errno))
        if fd is not None:
            os.close(fd)

    if not result["errors"] and result["vm_created"] and result["vm_destroyed"]:
        result["verdict"] = "PASS"
    return result


if __name__ == "__main__":
    result = probe()
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["verdict"] == "PASS" else 1)
