"""
verify_delay_distributions.py

Calibration backtest for the Monte Carlo path simulator (app/path_simulation.py):
samples real multi-leg rotations from flights_propagation, replays each
rotation's actual airport sequence and actual leg-1 departure delay through
simulate_path() (importing the real app module, not a reimplementation), and
checks what fraction of the time the ACTUAL observed arrival delay at each
subsequent stop falls inside the simulated [p10, p90] band.

Downstream hops are NOT reseeded with the rotation's real observed delays —
only leg 1's actual DepDelayMinutes seeds the simulation; every hop after
that runs on the simulator's own chained draws, same as a real user's
hypothetical path. This is what makes the backtest a genuine test of
simulate_path()'s forward-chaining behavior, not just a one-hop lookup check.

A well-calibrated p10-p90 band should cover the true outcome ~80% of the
time. Broken down by hop position (does calibration degrade over a longer
path?) and by the starting-delay bucket (is any one inbound-delay regime
badly off?) to catch a narrower miscalibration that an aggregate number
would hide.

Run after build_delay_distributions.py. Read-only — writes nothing to the DB.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from path_simulation import simulate_path, bucket_for, N_DRAWS_DEFAULT  # noqa: E402

DB_PATH = Path("flights.duckdb")

# ~200 rotations balances backtest runtime against having enough
# hop-observations (a rotation with k legs contributes k-1 checks) to trust
# the per-bucket/per-hop breakdown, not just the aggregate.
N_ROTATIONS = 200


def load_lookups(con: duckdb.DuckDBPyConnection):
    node_df = con.execute("SELECT * FROM node_delay_distributions").df()
    edge_df = con.execute("SELECT * FROM edge_delay_distributions").df()
    net_node_df = con.execute("SELECT * FROM network_node_delay_distributions").df()
    net_edge_df = con.execute("SELECT * FROM network_edge_delay_distributions").df()
    node_lookup = {(r.airport, r.inbound_bucket): np.asarray(r.dep_delay_samples) for r in node_df.itertuples()}
    edge_lookup = {(r.origin, r.dest, r.dep_bucket): np.asarray(r.arr_delay_samples) for r in edge_df.itertuples()}
    network_node_lookup = {r.inbound_bucket: np.asarray(r.dep_delay_samples) for r in net_node_df.itertuples()}
    network_edge_lookup = {r.dep_bucket: np.asarray(r.arr_delay_samples) for r in net_edge_df.itertuples()}
    return node_lookup, edge_lookup, network_node_lookup, network_edge_lookup


def sample_rotations(con: duckdb.DuckDBPyConnection, n: int):
    # Rotations need >=2 legs: leg 1 seeds the simulation's starting
    # condition, and there must be at least one subsequent leg to check
    # calibration against.
    return con.execute(f"""
        SELECT Tail_Number, operating_day
        FROM flights_propagation
        WHERE Tail_Number IS NOT NULL AND operating_day IS NOT NULL
        GROUP BY Tail_Number, operating_day
        HAVING COUNT(*) >= 2
        ORDER BY random()
        LIMIT {n}
    """).fetchall()


def rotation_legs(con: duckdb.DuckDBPyConnection, tail: str, day):
    return con.execute("""
        SELECT Origin, Dest, DepDelayMinutes, ArrDelayMinutes
        FROM flights_propagation
        WHERE Tail_Number = ? AND operating_day = ?
        ORDER BY rotation_leg_number
    """, [tail, day]).fetchall()


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
    required = {"flights_propagation", "node_delay_distributions", "edge_delay_distributions",
                "network_node_delay_distributions", "network_edge_delay_distributions"}
    missing = required - tables
    if missing:
        raise RuntimeError(f"Missing table(s) {missing}. Run build_propogation.py and "
                            f"build_delay_distributions.py first.")

    node_lookup, edge_lookup, network_node_lookup, network_edge_lookup = load_lookups(con)
    known_airports = {a for a, _ in node_lookup}

    rotations = sample_rotations(con, N_ROTATIONS)

    records = []  # (hop_position, start_bucket, covered)
    skipped = 0
    for tail, day in rotations:
        legs = rotation_legs(con, tail, day)
        if any(leg[2] is None or leg[3] is None for leg in legs):
            skipped += 1
            continue

        path = [legs[0][0]] + [leg[1] for leg in legs]
        if any(a not in known_airports for a in path[1:-1]):
            # An intermediate stop never seen as a connecting airport in
            # node_delay_distributions — shouldn't happen for a REAL
            # rotation's connecting airports, but skip defensively rather
            # than crash the backtest on a data oddity.
            skipped += 1
            continue

        leg1_dep_delay = legs[0][2]
        actual_arr_delays = [leg[3] for leg in legs]
        start_bucket = bucket_for(leg1_dep_delay)

        results = simulate_path(
            path, leg1_dep_delay,
            node_lookup, edge_lookup, network_node_lookup, network_edge_lookup,
            n_draws=N_DRAWS_DEFAULT,
        )
        for hop_position, (r, actual) in enumerate(zip(results, actual_arr_delays), start=1):
            p10 = np.percentile(r.arr_delay_draws, 10)
            p90 = np.percentile(r.arr_delay_draws, 90)
            records.append((hop_position, start_bucket, p10 <= actual <= p90))

    if not records:
        raise RuntimeError("No usable rotations sampled — check flights_propagation has multi-leg rotations.")

    covered = sum(1 for _, _, c in records if c)
    print(f"Sampled {len(rotations)} rotations ({skipped} skipped for missing/unusable data), "
          f"{len(records)} hop-observations.")
    print(f"Overall coverage: {covered / len(records):.1%} ({covered}/{len(records)}) "
          f"— target ~80% for a well-calibrated [p10, p90] band.")

    print("\nBy hop position:")
    max_hop = max(h for h, _, _ in records)
    for hop in range(1, max_hop + 1):
        subset = [c for h, _, c in records if h == hop]
        if subset:
            print(f"  hop {hop}: {sum(subset) / len(subset):.1%} (n={len(subset)})")

    print("\nBy starting-delay bucket:")
    for bucket in ["on_time_or_early", "1_15", "16_30", "31_60", "60_plus"]:
        subset = [c for _, b, c in records if b == bucket]
        if subset:
            print(f"  {bucket:<18} {sum(subset) / len(subset):.1%} (n={len(subset)})")

    con.close()


if __name__ == "__main__":
    main()
