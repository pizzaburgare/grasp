#!/usr/bin/env python3
"""Benchmark TTS throughput and memory usage across batch sizes.

This script measures for each batch size:
- Wall-clock synthesis time
- Total generated clip duration
- Real-time factor (RTF = wall_time / clip_duration)
- Speed vs realtime (clip_duration / wall_time)
- Peak process memory (RSS)

Peak memory is measured in a subprocess per batch size so values are isolated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.core.settings import DEFAULT_TTS_ENGINE
from src.tts import available_engines, get_default_engine

_SAMPLE_TEXTS = [
    "Queueing systems are often modelled with stochastic processes.",
    "The Erlang B formula gives blocking probability in loss systems.",
    "Laplace transforms are used for continuous-time signal analysis.",
    "Z transforms serve a similar role for discrete-time systems.",
    "Little's law states that L equals lambda times W.",
    "Markov chains can represent state transitions over time.",
    "Service rate and arrival rate determine utilization.",
    "Stability requires average service capacity above offered load.",
]


@dataclass(slots=True)
class BenchmarkResult:
    batch_size: int
    clips: int
    wall_time_seconds: float
    clip_time_seconds: float
    real_time_factor: float
    speed_vs_realtime: float
    load_time_seconds: float
    peak_rss_mib: float
    accelerator_peak_mib: float | None


def _parse_batch_sizes(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"Invalid batch size: {value}. Must be >= 1")
        values.append(value)

    if not values:
        raise ValueError("No batch sizes provided")
    return values


def _make_texts(num_clips: int) -> list[str]:
    texts: list[str] = []
    for i in range(num_clips):
        base = _SAMPLE_TEXTS[i % len(_SAMPLE_TEXTS)]
        texts.append(f"[{i + 1}] {base}")
    return texts


def _ru_maxrss_mib() -> float:
    """Return peak RSS (MiB) for this process.

    macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


class _AcceleratorTracker:
    def __init__(self) -> None:
        self.peak_mib: float | None = None
        self._torch: Any | None = None
        self._kind: str | None = None

        try:
            import torch  # pylint: disable=import-outside-toplevel

            self._torch = torch
        except ImportError:
            return

        torch = self._torch
        if torch.cuda.is_available():
            self._kind = "cuda"
            torch.cuda.reset_peak_memory_stats()
            return

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._kind = "mps"

    def sample(self) -> None:
        if self._torch is None or self._kind is None:
            return

        if self._kind == "cuda":
            bytes_used = float(self._torch.cuda.max_memory_allocated())
            mib = bytes_used / (1024 * 1024)
        else:
            mps = getattr(self._torch, "mps", None)
            if mps is None or not hasattr(mps, "driver_allocated_memory"):
                return
            bytes_used = float(mps.driver_allocated_memory())
            mib = bytes_used / (1024 * 1024)

        if self.peak_mib is None:
            self.peak_mib = mib
        else:
            self.peak_mib = max(self.peak_mib, mib)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        default=os.environ.get("TTS_ENGINE", DEFAULT_TTS_ENGINE),
        help="TTS engine to use (qwen|kokoro|piper)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes to benchmark",
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=16,
        help="Number of clips to synthesize per benchmark run",
    )
    parser.add_argument(
        "--warmup-clips",
        type=int,
        default=1,
        help="Number of warmup clips before timing",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to run each batch size",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable benchmark JSON",
    )

    # Internal child mode for isolated memory measurements.
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-run-index", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def _load_engine(engine_name: str) -> tuple[Any, float]:
    start = time.perf_counter()
    os.environ["TTS_ENGINE"] = engine_name
    engine = get_default_engine()

    for loader in ("_load_pipeline", "_load_model", "_load_voice"):
        maybe = getattr(engine, loader, None)
        if callable(maybe):
            maybe()
            break

    return engine, time.perf_counter() - start


def _run_single_benchmark(
    engine_name: str,
    batch_size: int,
    clips: int,
    warmup_clips: int,
) -> BenchmarkResult:
    engine, load_time = _load_engine(engine_name)
    tracker = _AcceleratorTracker()

    texts = _make_texts(clips)
    warmup = min(max(warmup_clips, 0), clips)

    for text in texts[:warmup]:
        engine.synthesize(text)
        tracker.sample()

    timed_texts = texts[warmup:]
    total_clip_seconds = 0.0

    start = time.perf_counter()
    for i in range(0, len(timed_texts), batch_size):
        chunk = timed_texts[i : i + batch_size]
        outputs = engine.synthesize_batch(chunk)
        if len(outputs) != len(chunk):
            raise RuntimeError(
                f"Expected {len(chunk)} outputs from synthesize_batch, got {len(outputs)}"
            )

        for audio, sample_rate in outputs:
            total_clip_seconds += len(audio) / sample_rate
        tracker.sample()

    wall_time = time.perf_counter() - start

    if total_clip_seconds <= 0.0:
        real_time_factor = math.inf
        speed_vs_realtime = 0.0
    else:
        real_time_factor = wall_time / total_clip_seconds
        speed_vs_realtime = total_clip_seconds / wall_time

    return BenchmarkResult(
        batch_size=batch_size,
        clips=len(timed_texts),
        wall_time_seconds=wall_time,
        clip_time_seconds=total_clip_seconds,
        real_time_factor=real_time_factor,
        speed_vs_realtime=speed_vs_realtime,
        load_time_seconds=load_time,
        peak_rss_mib=_ru_maxrss_mib(),
        accelerator_peak_mib=tracker.peak_mib,
    )


def _run_child(args: argparse.Namespace) -> int:
    if args.child_batch_size is None or args.child_run_index is None:
        raise ValueError("child mode requires --child-batch-size and --child-run-index")

    result = _run_single_benchmark(
        engine_name=args.engine,
        batch_size=args.child_batch_size,
        clips=args.clips,
        warmup_clips=args.warmup_clips,
    )
    payload = {
        "run_index": args.child_run_index,
        "result": asdict(result),
    }
    print(json.dumps(payload))
    return 0


def _spawn_child(
    args: argparse.Namespace,
    batch_size: int,
    run_index: int,
) -> BenchmarkResult:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--engine",
        args.engine,
        "--clips",
        str(args.clips),
        "--warmup-clips",
        str(args.warmup_clips),
        "--child",
        "--child-batch-size",
        str(batch_size),
        "--child-run-index",
        str(run_index),
    ]

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    line = completed.stdout.strip().splitlines()[-1]
    parsed = json.loads(line)
    data = parsed["result"]
    return BenchmarkResult(**data)


def _print_summary(engine_name: str, grouped: dict[int, list[BenchmarkResult]]) -> None:
    print("=" * 95)
    print(f"TTS Batch Size Benchmark [{engine_name}]")
    print("=" * 95)
    print(
        "batch  runs  clips/run  wall(s)   clip(s)   "
        "RTF(real/clip)  speed(x)  peak RSS(MiB)  accel(MiB)"
    )

    for batch_size in sorted(grouped):
        runs = grouped[batch_size]
        wall = statistics.mean(r.wall_time_seconds for r in runs)
        clip = statistics.mean(r.clip_time_seconds for r in runs)
        rtf = statistics.mean(r.real_time_factor for r in runs)
        speed = statistics.mean(r.speed_vs_realtime for r in runs)
        peak_rss = max(r.peak_rss_mib for r in runs)
        accel_values = [r.accelerator_peak_mib for r in runs if r.accelerator_peak_mib is not None]
        accel_peak = max(accel_values) if accel_values else None
        accel_col = f"{accel_peak:10.1f}" if accel_peak is not None else "         n/a"

        print(
            f"{batch_size:>5}  {len(runs):>4}  {runs[0].clips:>9}  {wall:>7.2f}  {clip:>8.2f}"
            f"  {rtf:>14.3f}  {speed:>8.3f}  {peak_rss:>13.1f}  {accel_col}"
        )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    engines = available_engines()
    args.engine = str(args.engine).lower()
    if args.engine not in engines:
        parser.error(f"Unknown engine '{args.engine}'. Choose one of: {', '.join(engines)}")

    if args.clips < 1:
        parser.error("--clips must be >= 1")
    if args.warmup_clips < 0:
        parser.error("--warmup-clips must be >= 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    if args.child:
        return _run_child(args)

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    grouped_results: dict[int, list[BenchmarkResult]] = {}

    for batch_size in batch_sizes:
        grouped_results[batch_size] = []
        for run_index in range(args.repeats):
            result = _spawn_child(args, batch_size=batch_size, run_index=run_index)
            grouped_results[batch_size].append(result)

    _print_summary(args.engine, grouped_results)

    if args.json_out is not None:
        payload: dict[str, Any] = {
            "engine": args.engine,
            "batch_sizes": batch_sizes,
            "clips": args.clips,
            "warmup_clips": args.warmup_clips,
            "repeats": args.repeats,
            "results": {
                str(batch): [asdict(r) for r in runs] for batch, runs in grouped_results.items()
            },
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON results to: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
