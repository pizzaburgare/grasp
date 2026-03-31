import io
from unittest.mock import patch

from src.core.command_runner import run_command

EXPECTED_THREAD_COUNT = 2


class _StuckProcess:
    def __init__(self) -> None:
        self.stdout: io.StringIO = io.StringIO("")
        self.stderr: io.StringIO = io.StringIO("")
        self.returncode: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        return self.returncode if self.returncode is not None else -9


class _ImmediateProcess:
    def __init__(self) -> None:
        self.stdout: io.StringIO = io.StringIO("")
        self.stderr: io.StringIO = io.StringIO("")
        self.returncode: int | None = 0

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        return self.returncode if self.returncode is not None else -9


class _ThreadSpy:
    def __init__(self, target: object, args: tuple[object, ...], daemon: bool) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon
        self.join_calls: list[float | None] = []

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def is_alive(self) -> bool:
        return False


class TestCommandRunnerSafety:
    def test_watchdog_kills_hung_subprocess(self) -> None:
        process = _StuckProcess()
        with patch("src.core.command_runner.subprocess.Popen", return_value=process):
            result = run_command(
                command=["fake", "cmd"],
                env={},
                status_label="Compiling video",
                poll_interval_seconds=0.001,
                watchdog_timeout_seconds=0.01,
                drain_join_timeout_seconds=0.1,
            )

        assert process.killed is True
        assert result.returncode != 0
        assert "timed out" in result.stderr

    def test_drain_threads_join_with_timeout(self) -> None:
        process = _ImmediateProcess()
        created_threads: list[_ThreadSpy] = []

        def _thread_factory(
            *, target: object, args: tuple[object, ...], daemon: bool
        ) -> _ThreadSpy:
            thread = _ThreadSpy(target=target, args=args, daemon=daemon)
            created_threads.append(thread)
            return thread

        with (
            patch("src.core.command_runner.subprocess.Popen", return_value=process),
            patch("src.core.command_runner.threading.Thread", side_effect=_thread_factory),
        ):
            run_command(
                command=["fake", "cmd"],
                env={},
                status_label="Compiling video",
                drain_join_timeout_seconds=12.5,
                watchdog_timeout_seconds=10.0,
            )

        assert len(created_threads) == EXPECTED_THREAD_COUNT
        assert all(thread.join_calls == [12.5] for thread in created_threads)
