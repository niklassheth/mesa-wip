# SPDX-License-Identifier: MIT
"""Shared native-index log policy: retain bounded polls as diagnostics.
(v3, 2026-10-24: co-processor crashes are hard faults.)

Match the regression guard's existing positive-retirement, sub-second limit.
Only a complete notification-poll message is exempted; hard events win and
unknown/malformed/zero-retirement/long timeout messages remain faults.
"""
import math
import re

POLL_CEILING = 32

_POLL = re.compile(
    r'(?:\[\s*\d+\.\d+\]\s+)?(?:asahi_m4: )?'
    r'G16: notification wait timed out; polling retired ([1-9][0-9]*) '
    r'Work\(s\), oldest flight ([0-9]{1,3}) ms', re.I)
_HARD = re.compile(
    r'retirement watchdog|(?:firmware|IOMMU|MMU).*\bfault\b|'
    # v3: co-processor crashes are hard faults (G16 firmware; any RTKit co-processor: announcement,
    # dump headers, crashlog errors). v2 caught tonight's GPU crashes only through a following
    # "submission pipeline failed: EIO" line. Deliberately not "G16: crash buffer" (probe-time
    # allocation) or "RTKit: reused crashlog buffer" / "Version:" / "Message (id=...)".
    r'G16: firmware crashed|RTKit: (?:co-processor has crashed|Device is crashed|Async error info|'
    r'Exception dump:|Crash time:|Unknown crashlog|Invalid crashlog|Expected crashlog header|'
    r'Crashlog size \S+ is too large|End of crashlog reached but no footer|timed out waiting for crash buffer)|'
    r'G16.*(?:closing a client|drain)|didn.t complete|leaking|'
    r'\b(?:Oops|SError)\b|(?-i:\b(?:BUG|WARN|WARNING)\b)|Kernel panic', re.I)
_CMDLINE = re.compile(r'(?:\[\s*\d+\.\d+\]\s+)?Kernel command line: [^\r\n]*')
_FAULT = re.compile(
    r'ETIMEDOUT|\bEIO\b|hang flight|'
    r'G16.*(?:fault|timeout|timed out|hang)|'
    r'GPU.*(?:fault|error|halt|timed out|timeout|reset)|'
    r'(?:asahi|AGX).*(?:fault|timed out|timeout|halted|unrecoverable|resetting)|'
    r'G16: command \d+ preparation failed:', re.I)


def notification_poll_diagnostic(line):
    """Return parsed evidence only for the entire bounded poll line."""
    line = line.rstrip('\r\n')
    if _HARD.search(line):
        return None
    match = _POLL.fullmatch(line)
    if not match:
        return None
    return dict(kind='bounded_positive_notification_poll', line=line,
                retired_works=int(match[1]), oldest_flight_ms=int(match[2]))


def classify_kernel_line(line):
    # This boot announcement quotes options such as asahi_m4.soft_faults=1;
    # it is not an event. The existing regression monitor makes this exclusion.
    if _CMDLINE.fullmatch(line.rstrip('\r\n')):
        return 'other'
    if _HARD.search(line):
        return 'fault'
    if notification_poll_diagnostic(line) is not None:
        return 'diagnostic'
    return 'fault' if _FAULT.search(line) else 'other'


def scan_kernel_log(text):
    faults, diagnostics = [], []
    for line in text.splitlines():
        kind = classify_kernel_line(line)
        if kind == 'fault':
            faults.append(line)
        elif kind == 'diagnostic':
            diagnostics.append(notification_poll_diagnostic(line))
    return dict(faults=faults, diagnostics=diagnostics)


def notification_poll_window(before, after):
    """Account only new lines on one boot, with a rate over the full window.

    A wrapped log cannot prove the count; preserve its visible diagnostics and
    report incomplete coverage instead of treating the smaller count as a pass.
    """
    old = before['kernel_log']['output']
    new = after['kernel_log']['output']
    same_boot = before['boot_id'] == after['boot_id']
    covered = same_boot and new.startswith(old)
    lines = new[len(old):] if covered else new
    count = len(scan_kernel_log(lines)['diagnostics'])
    start, end = before.get('boot_age_seconds'), after.get('boot_age_seconds')
    valid_time = all(type(t) in (int, float) and math.isfinite(t) and t >= 0
                     for t in (start, end))
    elapsed = end - start if valid_time and same_boot else None
    valid_time = elapsed is not None and (elapsed > 0 or elapsed == count == 0)
    rate = (60 * count / elapsed if elapsed else 0.0) if valid_time else None
    return dict(count=count, ceiling=POLL_CEILING, exceeded=count > POLL_CEILING,
                window_seconds=elapsed, rate_per_minute=rate,
                complete_log_coverage=covered, rate_available=valid_time,
                observation_complete=covered and valid_time)
