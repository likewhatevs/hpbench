#!/usr/bin/env python3
"""Perf-based validation for hpbench TLB statistics.

The script runs hpbench with reduced thread/iteration counts and cross-checks
its aggregated ITLB/DTLB counters against `perf stat` readings.  It expects a
binary path via --binary (default: ./hpbench) and honours the PERF and SUDO
environment variables to customise tooling/privilege.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


DEFAULT_ITERATIONS = "64"
BASE_HP_ARGS = (
    "--threads",
    "2",
    "--iterations",
    DEFAULT_ITERATIONS,
    "--size",
    "4194304",
)


@dataclass(frozen=True)
class Scenario:
    """Definition of one hpbench scenario exercised by the validator.

    Attributes:
        name: Human-readable scenario label used in output.
        extra_args: Tuple of CLI fragments that isolate this benchmark mode.
    """

    name: str
    extra_args: Tuple[str, ...]


ITLB_PATTERN = re.compile(
    r"ITLB:\s+(?P<accesses>\d+) accesses,\s+(?P<hits>\d+) hits,\s+(?P<misses>\d+) misses"
)
DTLB_PATTERN = re.compile(
    r"DTLB:\s+(?P<accesses>\d+) accesses,\s+(?P<hits>\d+) hits,\s+(?P<misses>\d+) misses"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default="./hpbench",
        help="Path to the hpbench binary (default: ./hpbench)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Relative tolerance allowed when comparing counters (default: 10%%)",
    )
    parser.add_argument(
        "--perf",
        default=os.environ.get("PERF", "perf"),
        help="perf executable to use (default: $PERF or 'perf')",
    )
    parser.add_argument(
        "--sudo",
        default=os.environ.get("SUDO", ""),
        help="Optional privilege escalation command (default: $SUDO)",
    )
    parser.add_argument(
        "--skip-code",
        action="store_true",
        help="Skip code benchmarks when running hpbench/perf",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data benchmarks when running hpbench/perf",
    )
    parser.add_argument(
        "--skip-code-4k",
        dest="skip_code_4k",
        action="store_true",
        help="Skip the 4K-page code benchmark",
    )
    parser.add_argument(
        "--skip-code-regular",
        dest="skip_code_4k",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-code-huge",
        action="store_true",
        help="Skip the huge-page code benchmark",
    )
    parser.add_argument(
        "--skip-data-4k",
        dest="skip_data_4k",
        action="store_true",
        help="Skip the 4K-page data benchmark",
    )
    parser.add_argument(
        "--skip-data-regular",
        dest="skip_data_4k",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-data-huge",
        action="store_true",
        help="Skip the huge-page data benchmark",
    )
    return parser.parse_args()


def _run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def _maybe_split_sudo(sudo_value: str) -> Tuple[str, ...]:
    sudo_value = sudo_value.strip()
    if not sudo_value:
        return tuple()
    return tuple(shlex.split(sudo_value))


def _build_hpbench_cmd(
    binary: str,
    *,
    disable_perf: bool,
    extra_args: Sequence[str],
) -> Tuple[str, ...]:
    cmd = [binary, *BASE_HP_ARGS]
    if disable_perf:
        cmd.append("--no-perf")
    cmd.extend(extra_args)
    return tuple(cmd)


def _run_hpbench(
    binary: str,
    *,
    disable_perf: bool = False,
    extra_args: Sequence[str] = (),
) -> Tuple[Dict[str, int], Dict[str, int], str]:
    cmd = _build_hpbench_cmd(binary, disable_perf=disable_perf, extra_args=extra_args)

    result = _run_command(cmd)
    itlb_totals, dtlb_totals = _accumulate_counters(result.stdout)
    return itlb_totals, dtlb_totals, result.stdout


def _accumulate_counters(text: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    itlb_totals = {"accesses": 0, "hits": 0, "misses": 0}
    dtlb_totals = {"accesses": 0, "hits": 0, "misses": 0}

    for match in ITLB_PATTERN.finditer(text):
        accesses = int(match.group("accesses"))
        hits = int(match.group("hits"))
        misses = int(match.group("misses"))
        itlb_totals["accesses"] += accesses
        itlb_totals["hits"] += hits
        itlb_totals["misses"] += misses

    for match in DTLB_PATTERN.finditer(text):
        accesses = int(match.group("accesses"))
        hits = int(match.group("hits"))
        misses = int(match.group("misses"))
        dtlb_totals["accesses"] += accesses
        dtlb_totals["hits"] += hits
        dtlb_totals["misses"] += misses

    return itlb_totals, dtlb_totals


_AMD_EVENT_SET = (
    ("bp_l1_tlb_fetch_hit", "bp_l1_tlb_fetch_hit:u"),
    ("bp_l1_tlb_miss_l2_hit", "bp_l1_tlb_miss_l2_hit:u"),
    ("bp_l1_tlb_miss_l2_tlb_miss", "bp_l1_tlb_miss_l2_tlb_miss:u"),
    ("dTLB-loads", "dTLB-loads:u"),
    ("dTLB-load-misses", "dTLB-load-misses:u"),
)


def _scenario_matrix(args: argparse.Namespace) -> Tuple[Scenario, ...]:
    """Build the ordered list of benchmark scenarios to execute.

    The returned tuple is filtered according to the caller's skip flags so
    that perf validation only exercises the requested workloads.
    """
    definitions = (
        Scenario("code-4K", ("--skip-data", "--skip-code-huge")),
        Scenario("code-hugetlb-2MB", ("--skip-data", "--skip-code-4k")),
        Scenario("data-4K", ("--skip-code", "--skip-data-huge")),
        Scenario("data-hugetlb-1GB", ("--skip-code", "--skip-data-4k")),
    )

    enabled: list[Scenario] = []
    for scenario in definitions:
        if scenario.name == "code-4K" and (args.skip_code or args.skip_code_4k):
            continue
        if scenario.name == "code-hugetlb-2MB" and (args.skip_code or args.skip_code_huge):
            continue
        if scenario.name == "data-4K" and (args.skip_data or args.skip_data_4k):
            continue
        if scenario.name == "data-hugetlb-1GB" and (args.skip_data or args.skip_data_huge):
            continue
        enabled.append(scenario)
    return tuple(enabled)


def _run_perf(perf: str, sudo: Tuple[str, ...], binary: str,
              *, extra_args: Sequence[str]) -> Tuple[Dict[str, int], Optional[str]]:
    # perf understands the symbolic event names directly; no extra prefix required.
    counts = _invoke_perf(perf, sudo, binary, _AMD_EVENT_SET,
                          extra_args=extra_args)
    if counts is None:
        print("perf stat could not collect AMD ITLB events; ensure raw counters are supported and accessible", file=sys.stderr)
        sys.exit(1)
    if not counts:
        return {}, None
    return counts, "AMD"


def _build_perf_cmd(
    perf: str,
    sudo: Tuple[str, ...],
    binary: str,
    events: Iterable[Tuple[str, str]],
    *,
    extra_args: Sequence[str],
) -> Tuple[str, ...]:
    cmd = list(sudo) + [perf, "stat", "-x,"]
    cmd.extend(f"-e{spec}" for _, spec in events)
    cmd.extend(_build_hpbench_cmd(binary, disable_perf=True, extra_args=extra_args))
    return tuple(cmd)


def _invoke_perf(perf: str, sudo: Tuple[str, ...], binary: str,
                 events: Iterable[Tuple[str, str]],
                 *,
                 extra_args: Sequence[str]) -> Optional[Dict[str, int]]:
    cmd = _build_perf_cmd(perf, sudo, binary, events, extra_args=extra_args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("perf executable not found; skipping validation", file=sys.stderr)
        return {}

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        lower = stderr.lower()
        if "permission" in lower or "paranoid" in lower:
            print("perf stat requires elevated permissions; skipping validation", file=sys.stderr)
            return {}
        if proc.returncode == 129 or "invalid" in lower or "not supported" in lower or "unknown" in lower:
            # try next event set
            return None
        proc.check_returncode()

    counts: Dict[str, int] = {}
    event_lookup: Dict[str, str] = {}
    for label, spec in events:
        event_lookup[spec] = label
        base = spec.split(':', 1)[0]
        event_lookup.setdefault(base, label)
    for line in proc.stderr.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        value_str = parts[0].strip()
        event_name = parts[2].strip()
        if not value_str or value_str == "<not supported>" or not event_name:
            continue
        try:
            value = int(value_str)
        except ValueError:
            continue
        base_name = event_name.split(':', 1)[0]
        label = event_lookup.get(event_name) or event_lookup.get(base_name)
        if label:
            counts[label] = value
    # ensure we captured at least one requested counter
    if not counts:
        return None
    return counts


def _validate(
    itlb_prog: Dict[str, int],
    dtlb_prog: Dict[str, int],
    perf_counts: Dict[str, int],
    perf_mode: Optional[str],
    tolerance: float,
    *,
    check_itlb: bool,
    check_dtlb: bool,
    context: Optional[str] = None,
) -> None:
    failed = False

    actuals = _materialise_actuals(perf_counts, perf_mode)

    if check_itlb:
        failed |= not _report_event(
            "ITLB accesses",
            itlb_prog["accesses"],
            actuals.get("ITLB accesses"),
            tolerance,
            context=context,
        )
        failed |= not _report_event(
            "ITLB misses",
            itlb_prog["misses"],
            actuals.get("ITLB misses"),
            tolerance,
            context=context,
        )

    if check_dtlb:
        failed |= not _report_event(
            "DTLB accesses",
            dtlb_prog["accesses"],
            actuals.get("DTLB accesses"),
            tolerance,
            context=context,
        )
        failed |= not _report_event(
            "DTLB misses",
            dtlb_prog["misses"],
            actuals.get("DTLB misses"),
            tolerance,
            context=context,
        )

    if failed:
        raise AssertionError("perf validation mismatch")


def _materialise_actuals(perf_counts: Dict[str, int], mode: Optional[str]) -> Dict[str, int]:
    if not perf_counts or mode is None:
        return {}

    if mode == "AMD":
        hits = (
            perf_counts.get("bp_l1_tlb_fetch_hit", 0)
            + perf_counts.get("bp_l1_tlb_miss_l2_hit", 0)
        )
        misses = perf_counts.get("bp_l1_tlb_miss_l2_tlb_miss", 0)
        accesses = hits + misses
    else:
        accesses = perf_counts.get("itlb_access", 0)
        misses = perf_counts.get("itlb_miss", 0)
        hits = accesses - misses if accesses is not None and misses is not None else 0

    actuals = {
        "ITLB accesses": accesses,
        "ITLB misses": misses,
        "DTLB accesses": perf_counts.get("dTLB-loads"),
        "DTLB misses": perf_counts.get("dTLB-load-misses"),
    }
    return actuals


def _report_event(
    label: str,
    expected: int,
    actual: Optional[int],
    tolerance: float,
    *,
    context: Optional[str] = None,
) -> bool:
    if actual is None:
        prefix = f"{context} " if context else ""
        print(f"SKIP: {prefix}{label}: hpbench={expected} perf=NA reason=no-perf-data")
        return True

    diff = abs(expected - actual)

    limit = max(1000, int(math.ceil(max(expected, 1) * tolerance)))
    status = "PASS" if diff <= limit else "FAIL"
    prefix = f"{context} " if context else ""
    denom = max(expected, 1)
    diff_pct = (diff / denom) * 100.0
    print(
        f"{status}: {prefix}{label}: hpbench={expected} perf={actual} diff={diff} "
        f"tol={limit} pct={diff_pct:.3f}%"
    )
    return diff <= limit


def main() -> int:
    args = _parse_args()
    sudo = _maybe_split_sudo(args.sudo)

    scenarios = _scenario_matrix(args)

    if not scenarios:
        print("No benchmarks selected for validation", file=sys.stderr)
        return 0

    for scenario in scenarios:
        itlb_prog, dtlb_prog, _ = _run_hpbench(
            args.binary,
            disable_perf=False,
            extra_args=scenario.extra_args,
        )
        perf_counts, perf_mode = _run_perf(
            args.perf,
            sudo,
            args.binary,
            extra_args=scenario.extra_args,
        )

        print(f"perf events mode ({scenario.name}): {perf_mode or 'unavailable'}")
        _validate(
            itlb_prog,
            dtlb_prog,
            perf_counts,
            perf_mode,
            args.tolerance,
            check_itlb=bool(itlb_prog["accesses"]),
            check_dtlb=bool(dtlb_prog["accesses"]),
            context=scenario.name,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
