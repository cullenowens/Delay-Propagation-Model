"""
path_simulation.py

Empirical conditional Monte Carlo engine for the "Flight Path Simulator"
tab in app.py: given a hypothetical multi-leg path (e.g. ATL -> SAF -> LAX
-> LGA — not necessarily a real tail's rotation) and a starting condition,
predict a delay DISTRIBUTION at each stop by resampling from real historical
outcomes, bucketed by inbound-delay size and airport/route identity.

Lives in app/ (not import_processing/) because it must be importable by the
deployed app, which is a self-contained folder with no shared packaging. It
takes plain dict/array arguments rather than owning a DB connection, so it's
independently testable.

DELAY_BUCKETS is duplicated from import_processing/build_delay_distributions.py
on purpose — app/ must not import import_processing/.

ALGORITHM, per call, vectorized by bucket group (<=5 rng.choice calls per
hop, so a full path at n_draws=2000 is effectively instant):
  1. dep_draws = n_draws copies of 0 (on-time) or the user's assumed
     starting delay.
  2. For each consecutive (origin, dest) pair in the path:
       - EDGE STEP: bucket each value in dep_draws (draws may already span
         multiple buckets from a prior hop); for each bucket group,
         resample from edge_lookup[(origin, dest, bucket)], falling back to
         network_edge_lookup[bucket] if that route/bucket combo is absent —
         the EXPECTED common case for a hypothetical path, not an error.
       - Record a StopResult for `dest`.
       - NODE STEP (if dest isn't the last stop): same group-and-resample
         against node_lookup[(dest, bucket)] / network_node_lookup[bucket],
         producing the next dep_draws.
  3. Return one StopResult per hop (len(path) - 1 total).

VALIDATION ASYMMETRY (important):
  - Airports are validated against the known node universe at input time —
    the UI rejects unknown codes immediately; there's no sensible fallback
    for "we've never seen this airport."
  - Routes may be entirely novel — that's the point of the feature. This
    module falls back to network_edge_lookup/network_node_lookup, never
    errors, and surfaces used_route_fallback / used_node_fallback so the UI
    can show a transparency note.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal

# Duplicated from build_delay_distributions.py on purpose (see module docstring).
DELAY_BUCKETS = [
    ("on_time_or_early", None, 0),
    ("1_15",              1,   15),
    ("16_30",             16,  30),
    ("31_60",             31,  60),
    ("60_plus",           61,  None),
]

N_DRAWS_DEFAULT = 2000


def bucket_for(delay_minutes: float) -> str:
    for label, low, high in DELAY_BUCKETS:
        if low is not None and delay_minutes < low:
            continue
        if high is not None and delay_minutes > high:
            continue
        return label
    raise ValueError(f"delay_minutes={delay_minutes!r} did not match any bucket")


@dataclass
class StopResult:
    airport: str
    origin: str
    arr_delay_draws: np.ndarray
    used_route_fallback: bool


def _resample_by_bucket(
    draws: np.ndarray,
    rng: np.random.Generator,
    pool_for_bucket,
    network_pool_for_bucket,
):
    """Buckets `draws`, resamples each bucket group from its own pool
    (falling back to the network-wide pool when the specific pool is
    missing/empty), and returns (result_draws, used_fallback) — result_draws
    is the same length/order as `draws`."""
    result = np.empty_like(draws, dtype=float)
    used_fallback = False
    buckets = np.array([bucket_for(d) for d in draws])
    for label, _, _ in DELAY_BUCKETS:
        mask = buckets == label
        n = int(mask.sum())
        if n == 0:
            continue
        pool = pool_for_bucket(label)
        if pool is None or len(pool) == 0:
            pool = network_pool_for_bucket(label)
            used_fallback = True
        result[mask] = rng.choice(pool, size=n, replace=True)
    return result, used_fallback


def simulate_path(
    path: list[str],
    start_condition: float | Literal["on_time"],
    node_lookup: dict[tuple[str, str], np.ndarray],
    edge_lookup: dict[tuple[str, str, str], np.ndarray],
    network_node_lookup: dict[str, np.ndarray],
    network_edge_lookup: dict[str, np.ndarray],
    n_draws: int = N_DRAWS_DEFAULT,
    seed: int | None = None,
) -> list[StopResult]:
    if len(path) < 2:
        raise ValueError("path must have at least 2 airports (an origin and a destination)")

    rng = np.random.default_rng(seed)
    start_delay = 0.0 if start_condition == "on_time" else float(start_condition)
    dep_draws = np.full(n_draws, start_delay, dtype=float)

    results: list[StopResult] = []
    for i in range(len(path) - 1):
        origin, dest = path[i], path[i + 1]

        arr_draws, used_route_fallback = _resample_by_bucket(
            dep_draws, rng,
            pool_for_bucket=lambda label: edge_lookup.get((origin, dest, label)),
            network_pool_for_bucket=lambda label: network_edge_lookup.get(label),
        )
        results.append(StopResult(
            airport=dest, origin=origin,
            arr_delay_draws=arr_draws,
            used_route_fallback=used_route_fallback,
        ))

        is_last_stop = (i == len(path) - 2)
        if not is_last_stop:
            dep_draws, _ = _resample_by_bucket(
                arr_draws, rng,
                pool_for_bucket=lambda label: node_lookup.get((dest, label)),
                network_pool_for_bucket=lambda label: network_node_lookup.get(label),
            )

    return results


def summarize(result: StopResult, thresholds=(15, 30, 60)) -> dict:
    draws = result.arr_delay_draws
    out = {
        "airport": result.airport,
        "origin": result.origin,
        "p10": float(np.percentile(draws, 10)),
        "p50": float(np.percentile(draws, 50)),
        "p90": float(np.percentile(draws, 90)),
        "mean": float(np.mean(draws)),
        "used_route_fallback": result.used_route_fallback,
    }
    for t in thresholds:
        out[f"prob_exceed_{t}"] = float(np.mean(draws > t))
    return out
