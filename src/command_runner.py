import subprocess
import threading
import time
from typing import IO


class SpinnerReporter:
    """Render spinner progress updates for a running command."""

    _SPINNER = ("|", "/", "-", "\\")

    def __init__(self, status_label: str) -> None:
        self.status_label = status_label

    def render_tick(self, spinner_index: int, elapsed: float) -> None:
        print(
            (
                f"\r{self.status_label} "
                f"{self._SPINNER[spinner_index % len(self._SPINNER)]} "
                f"{elapsed:5.1f}s"
            ),
            end="",
            flush=True,
        )

    def render_done(self, elapsed: float) -> None:
        print(f"\r{self.status_label} ✓ {elapsed:5.1f}s")


_WATCHDOG_WAIT_AFTER_KILL_SECONDS = 5


def check_process_timeout(
    process: subprocess.Popen[str],
    start: float,
    timeout_seconds: float,
) -> tuple[bool, float]:
    """Detect and handle subprocess timeout conditions."""
    elapsed = time.perf_counter() - start
    if elapsed < timeout_seconds:
        return False, elapsed

    process.kill()
    try:
        process.wait(timeout=_WATCHDOG_WAIT_AFTER_KILL_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
    return True, elapsed


class ProcessStreamDrainer:
    """Own stream-drain threads and collected subprocess output."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("Failed to capture subprocess stdout/stderr streams.")

        self._stdout_pipe = process.stdout
        self._stderr_pipe = process.stderr
        self._stdout_chunks: list[str] = []
        self._stderr_chunks: list[str] = []
        self._stdout_thread = threading.Thread(
            target=self._drain,
            args=(self._stdout_pipe, self._stdout_chunks),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain,
            args=(self._stderr_pipe, self._stderr_chunks),
            daemon=True,
        )

    @staticmethod
    def _drain(pipe: IO[str], buf: list[str]) -> None:
        for chunk in iter(lambda: pipe.read(4096), ""):
            buf.append(chunk)

    def start(self) -> None:
        self._stdout_thread.start()
        self._stderr_thread.start()

    def finish(self, join_timeout_seconds: float) -> tuple[str, str, bool]:
        self._stdout_thread.join(timeout=join_timeout_seconds)
        self._stderr_thread.join(timeout=join_timeout_seconds)
        drain_thread_stuck = self._stdout_thread.is_alive() or self._stderr_thread.is_alive()
        self._stdout_pipe.close()
        self._stderr_pipe.close()
        return "".join(self._stdout_chunks), "".join(self._stderr_chunks), drain_thread_stuck


class CommandResultFormatter:
    """Create and summarize command execution results."""

    _WATCHDOG_RETURN_CODE = 124
    _FAILURE_TAIL_LINE_COUNT = 30

    @staticmethod
    def build(
        command: list[str],
        process: subprocess.Popen[str],
        stdout_text: str,
        stderr_text: str,
        status_label: str,
        timed_out: bool,
        drain_thread_stuck: bool,
        watchdog_timeout_seconds: float,
        drain_join_timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        return_code = process.returncode if process.returncode is not None else 1

        if timed_out:
            timeout_note = (
                f"{status_label} timed out after {watchdog_timeout_seconds:.1f}s. "
                "Subprocess was killed by watchdog."
            )
            stderr_text = f"{stderr_text}\n{timeout_note}".strip()
            return_code = (
                return_code if return_code != 0 else CommandResultFormatter._WATCHDOG_RETURN_CODE
            )

        if drain_thread_stuck:
            join_note = (
                f"{status_label} output drain thread did not stop within "
                f"{drain_join_timeout_seconds:.1f}s join timeout."
            )
            stderr_text = f"{stderr_text}\n{join_note}".strip()

        return subprocess.CompletedProcess(
            args=command,
            returncode=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    @staticmethod
    def print_failure_tail(status_label: str, result: subprocess.CompletedProcess[str]) -> None:
        if result.returncode == 0:
            return

        combined = (result.stderr or "").strip() or (result.stdout or "").strip()
        if not combined:
            return

        tail = "\n".join(combined.splitlines()[-CommandResultFormatter._FAILURE_TAIL_LINE_COUNT :])
        print(f"{status_label} failed. Output (last 30 lines):")
        print(tail)


# TODO: Use existing library for spinner
POLL_INTERVAL_SECONDS = 0.2
DRAIN_JOIN_TIMEOUT_SECONDS = 30.0
WATCHDOG_TIMEOUT_SECONDS = 1800.0


def run_command(
    command: list[str],
    env: dict[str, str],
    status_label: str,
    *,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
    drain_join_timeout_seconds: float = DRAIN_JOIN_TIMEOUT_SECONDS,
    watchdog_timeout_seconds: float = WATCHDOG_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run subprocess commands with timeout, spinner, and non-blocking output drains."""
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    drainer = ProcessStreamDrainer(process)
    drainer.start()
    spinner = SpinnerReporter(status_label)

    i = 0
    timed_out = False
    while process.poll() is None:
        timed_out, elapsed = check_process_timeout(process, start, watchdog_timeout_seconds)
        if timed_out:
            break

        spinner.render_tick(i, elapsed)
        time.sleep(poll_interval_seconds)
        i += 1

    stdout_text, stderr_text, drain_thread_stuck = drainer.finish(drain_join_timeout_seconds)

    elapsed = time.perf_counter() - start
    spinner.render_done(elapsed)
    result = CommandResultFormatter.build(
        command=command,
        process=process,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        status_label=status_label,
        timed_out=timed_out,
        drain_thread_stuck=drain_thread_stuck,
        watchdog_timeout_seconds=watchdog_timeout_seconds,
        drain_join_timeout_seconds=drain_join_timeout_seconds,
    )
    CommandResultFormatter.print_failure_tail(status_label, result)

    return result
