"""Dev-only: starts and stops the telemetry simulators as child processes so
  the demo UI can run the whole demo from one window. Each simulator runs with
  the same Python interpreter as the UI, from the repo root, with its output
  sent to a log file in a temporary directory — nothing is written into the
  repo. Stopping sends SIGINT (the same as Ctrl+C in a terminal), so each
  simulator exits through its own KeyboardInterrupt handler and prints its
  "Stopped after N …" line; one that doesn't exit in time is killed.
  """
from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SIMULATOR_SCRIPTS = {
    "power": REPO_ROOT / "tools" / "telemetry_sim" / "power_simulator.py",
    "network": REPO_ROOT / "tools" / "telemetry_sim" / "network_simulator.py",
}

_STOP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class SimulatorStatus:
    name: str
    running: bool
    last_line: str


class SimulatorManager:
    def __init__(self) -> None:
        self._log_dir = Path(tempfile.mkdtemp(prefix="kumbia-demo-sims-"))
        self._processes: dict[str, subprocess.Popen] = {}

    def start(self, name: str) -> bool:
        """Starts the named simulator. Returns False if it is already running."""
        script = _script_for(name)
        if self._is_running(name):
            return False
        log = self._log_path(name).open("w", encoding="utf-8")
        self._processes[name] = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()  # the child keeps its own handle to the file
        return True

    def stop(self, name: str) -> bool:
        """Stops the named simulator the way Ctrl+C would. Returns False if it wasn't running."""
        _script_for(name)
        process = self._processes.get(name)
        if process is None or process.poll() is not None:
            return False
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return True

    def stop_all(self) -> None:
        for name in SIMULATOR_SCRIPTS:
            self.stop(name)

    def status(self) -> list[SimulatorStatus]:
        return [
            SimulatorStatus(name=name, running=self._is_running(name), last_line=self._last_line(name))
            for name in SIMULATOR_SCRIPTS
        ]

    def _is_running(self, name: str) -> bool:
        process = self._processes.get(name)
        return process is not None and process.poll() is None

    def _log_path(self, name: str) -> Path:
        return self._log_dir / f"{name}.log"

    def _last_line(self, name: str) -> str:
        path = self._log_path(name)
        if not path.exists():
            return ""
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
        return next((line for line in reversed(lines) if line), "")


def _script_for(name: str) -> Path:
    script = SIMULATOR_SCRIPTS.get(name)
    if script is None:
        raise ValueError(f"Unknown simulator {name!r}; expected one of {sorted(SIMULATOR_SCRIPTS)}.")
    return script
