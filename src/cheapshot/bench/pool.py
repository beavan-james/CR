"""Thread pool for benchmark calls that stops early when calls keep failing.

On a subscription, hitting the usage limit makes every following call fail fast;
without a circuit breaker a run would burn through hundreds of instant failures.
Stopped work is simply not cached, so re-running the same command resumes it.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class PoolResult:
    done: int = 0
    failures: int = 0
    not_run: int = 0
    stop_reason: str | None = None


def run_pool(jobs: Iterable[tuple], fn: Callable, workers: int = 4, on_done=None,
             max_consecutive_failures: int = 5) -> PoolResult:
    """Run fn(*job) for each job with at most `workers` in flight; jobs are submitted
    lazily so nothing new starts once the failure streak trips the breaker."""
    res, streak = PoolResult(), 0
    pending = iter(list(jobs))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        in_flight = set()

        def refill():
            while res.stop_reason is None and len(in_flight) < max(1, workers):
                job = next(pending, None)
                if job is None:
                    return
                in_flight.add(ex.submit(fn, *job))

        refill()
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for f in finished:
                in_flight.discard(f)
                err = f.exception()
                if err is None:
                    res.done += 1
                    streak = 0
                else:
                    res.failures += 1
                    streak += 1
                    if streak >= max_consecutive_failures and res.stop_reason is None:
                        res.stop_reason = f"{streak} failures in a row; last: {str(err)[:300]}"
                if on_done:
                    on_done(None if err else f.result(), err)
            refill()
    res.not_run = sum(1 for _ in pending)
    return res
