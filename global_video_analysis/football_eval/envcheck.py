"""Is this Python environment the one the project is tested with?

Checks the Python version (3.12 - 3.14) and every package pinned in ``../constraints.txt`` against what is
installed. ``setup_env.py`` runs it after installing; the launcher prints its warnings at start.

    python -m football_eval.envcheck           # report; exit code 1 when something is off
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_PYTHON = ((3, 12), (3, 14))          # inclusive range; keep in sync with setup_env.py
RECOMMENDED_PYTHON = (3, 13)
CONSTRAINTS = Path(__file__).resolve().parents[2] / "constraints.txt"
# packages that are pinned but legitimately absent: only one detector runtime is installed, tools are optional
OPTIONAL = {"onnxruntime", "onnxruntime-gpu", "jupyter", "pytest"}
ONE_OF = [("onnxruntime", "onnxruntime-gpu")]


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_constraints(path: Optional[Path] = None) -> Dict[str, str]:
    """``name==version`` lines of constraints.txt -> {canonical name: version}."""
    pins: Dict[str, str] = {}
    p = Path(path or CONSTRAINTS)
    if not p.exists():
        return pins
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if m:
            pins[canonical(m.group(1))] = m.group(2)
    return pins


def python_ok(version: Tuple[int, int] = None) -> bool:
    v = tuple(version or sys.version_info[:2])
    return SUPPORTED_PYTHON[0] <= v <= SUPPORTED_PYTHON[1]


def installed_versions(names) -> Dict[str, Optional[str]]:
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def compare(pins: Dict[str, str], installed: Dict[str, Optional[str]]) -> List[str]:
    """Problems: a required pin missing, a version different from the pin, both detector runtimes at once."""
    problems = []
    for name, want in sorted(pins.items()):
        have = installed.get(name)
        if have is None:
            if name not in OPTIONAL:
                problems.append(f"{name} is not installed (pinned {want})")
        elif have != want:
            problems.append(f"{name} {have} installed, pinned {want}")
    for group in ONE_OF:
        both = [n for n in group if installed.get(n)]
        if len(both) > 1:
            problems.append(f"both {' and '.join(both)} are installed: keep one (setup_env.py fixes it)")
    return problems


def check(constraints: Optional[Path] = None) -> List[str]:
    problems = []
    if not python_ok():
        lo, hi = SUPPORTED_PYTHON
        problems.append(f"Python {sys.version.split()[0]} is not supported (use {lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}, "
                        f"{RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]} recommended)")
    if sys.maxsize <= 2 ** 32:
        problems.append("32-bit Python: install the 64-bit version (no wheels exist for the scientific packages)")
    pins = read_constraints(constraints)
    problems += compare(pins, installed_versions(pins))
    return problems


def main() -> int:
    problems = check()
    pins = read_constraints()
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    if not pins:
        print(f"(no constraints file at {CONSTRAINTS})")
    if problems:
        print("Environment differs from constraints.txt:\n  - " + "\n  - ".join(problems))
        print("Fix: run setup_env.py from the football_video_evaluation folder (start.command / start.bat do it).")
        return 1
    print(f"Environment matches constraints.txt ({len(pins)} pins).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
