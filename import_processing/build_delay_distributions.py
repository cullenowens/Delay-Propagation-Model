"""
build_delay_distributions.py

Builds empirical delay-distribution tables on top of `flights_propagation`,
for the Monte Carlo path simulator (app/path_simulation.py). Where
airport_stats/route_stats collapse a turn/route's effect to an AVERAGE, these
tables keep the full historical spread — resampled at simulation time to
produce a DISTRIBUTION at each stop of a hypothetical multi-leg path, not a
single point estimate.

Run after build_propogation.py, before build_map_export.py. Safe to
re-run — everything is CREATE OR REPLACE.

BUCKETING
---------
Every row is bucketed by inbound-delay size into 5 fixed-width buckets (see
DELAY_BUCKETS) — fixed-width, not quantile-based, so a bucket's meaning is
stable as new months are added; a quantile boundary would silently shift on
every re-run.

NODE vs EDGE tables mirror airport_stats/route_stats' source universes:
  - node_delay_distributions: same universe as airport_stats
    (flights_propagation WHERE node_effect IS NOT NULL AND is_overnight_turn
    = FALSE), bucketed by prev_arr_delay (the inbound delay), sampling
    DepDelayMinutes (the outcome at that stop).
  - edge_delay_distributions: same universe as route_stats (all of
    flights_propagation), bucketed by this leg's own DepDelayMinutes,
    sampling ArrDelayMinutes.

Each keyed table (node/edge) is a FULL GRID — every known airport/route x
every bucket, via CROSS JOIN — so downstream lookups never hit a missing
row. Thin combos are flagged is_fallback and their dep/arr_delay_samples
column is PRE-BLENDED with the matching network-wide row, so callers never
need their own fallback logic.

Reservoir sampling uses the same ROW_NUMBER() OVER (PARTITION BY ... ORDER BY
random()) + filter pattern already idiomatic in this codebase's CTE style;
samples are cast to INTEGER (BTS delays are whole minutes) to halve stored
bytes.
"""

from pathlib import Path
import duckdb

DB_PATH = Path("flights.duckdb")

# Fixed-width bins (not quantile bins) so a bucket's meaning is stable as
# new months are added — a quantile boundary would silently shift on every
# re-run. (label, low, high) — low/high are None for open-ended bounds.
DELAY_BUCKETS = [
    ("on_time_or_early", None, 0),
    ("1_15",              1,   15),
    ("16_30",             16,  30),
    ("31_60",             31,  60),
    ("60_plus",           61,  None),
]

# A resampled distribution needs more support than a single percentile
# estimate (MIN_TURN_SAMPLE_THRESHOLD=10 in build_propogation.py is fine for
# one number; too thin for a resampled histogram's shape). 30 is the
# standard "large enough" cutoff.
MIN_BUCKET_SAMPLE_THRESHOLD = 30

# Reservoir-sample cap per (key, bucket): Monte Carlo resampling doesn't need
# the full population — a few hundred real draws already reproduce the
# shape — and capping keeps flights.duckdb and the exported map_data.duckdb
# bounded regardless of how much data gets added later.
MAX_STORED_SAMPLES = 200


def _bucket_case_sql(column: str) -> str:
    """Builds the bucket CASE expression once, used identically for node and
    edge queries — this is what makes adding a future grouping key (e.g.
    hour-of-day) a matter of adding one more helper + GROUP BY key, not
    restructuring."""
    clauses = []
    for label, low, high in DELAY_BUCKETS:
        if low is None:
            clauses.append(f"WHEN {column} <= {high} THEN '{label}'")
        elif high is None:
            clauses.append(f"WHEN {column} >= {low} THEN '{label}'")
        else:
            clauses.append(f"WHEN {column} BETWEEN {low} AND {high} THEN '{label}'")
    return "CASE " + " ".join(clauses) + " END"


def _bucket_bounds_values_sql() -> str:
    """A VALUES table of (bucket, low, high) — joined in so every output row
    carries its bucket's numeric bounds without repeating the CASE logic."""
    rows = []
    for label, low, high in DELAY_BUCKETS:
        low_sql = "NULL" if low is None else low
        high_sql = "NULL" if high is None else high
        rows.append(f"('{label}', {low_sql}, {high_sql})")
    return "(VALUES " + ", ".join(rows) + ") AS bb(bucket, bucket_low, bucket_high)"


def build_network_node_delay_distributions(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
    CREATE OR REPLACE TABLE network_node_delay_distributions AS
    WITH base AS (
        SELECT
            {_bucket_case_sql('prev_arr_delay')} AS inbound_bucket,
            DepDelayMinutes
        FROM flights_propagation
        WHERE node_effect IS NOT NULL
          AND is_overnight_turn = FALSE
    ),
    ranked AS (
        SELECT
            inbound_bucket,
            CAST(DepDelayMinutes AS INTEGER) AS dep_delay_sample,
            ROW_NUMBER() OVER (PARTITION BY inbound_bucket ORDER BY random()) AS rn
        FROM base
    )
    SELECT
        bb.bucket AS inbound_bucket,
        bb.bucket_low,
        bb.bucket_high,
        COUNT(r.dep_delay_sample) AS network_sample_size,
        array_agg(r.dep_delay_sample) FILTER (WHERE r.rn <= {MAX_STORED_SAMPLES}) AS dep_delay_samples
    FROM {_bucket_bounds_values_sql()}
    LEFT JOIN ranked r ON r.inbound_bucket = bb.bucket
    GROUP BY bb.bucket, bb.bucket_low, bb.bucket_high
    """)


def build_node_delay_distributions(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
    CREATE OR REPLACE TABLE node_delay_distributions AS
    WITH base AS (
        SELECT
            prev_dest AS airport,
            {_bucket_case_sql('prev_arr_delay')} AS inbound_bucket,
            DepDelayMinutes
        FROM flights_propagation
        WHERE node_effect IS NOT NULL
          AND is_overnight_turn = FALSE
    ),
    ranked AS (
        SELECT
            airport,
            inbound_bucket,
            CAST(DepDelayMinutes AS INTEGER) AS dep_delay_sample,
            ROW_NUMBER() OVER (PARTITION BY airport, inbound_bucket ORDER BY random()) AS rn
        FROM base
    ),
    per_airport AS (
        SELECT
            airport,
            inbound_bucket,
            COUNT(dep_delay_sample) AS airport_sample_size,
            array_agg(dep_delay_sample) FILTER (WHERE rn <= {MAX_STORED_SAMPLES}) AS own_samples
        FROM ranked
        GROUP BY airport, inbound_bucket
    ),
    -- Full grid: every known airport (from airport_stats) x every bucket, so
    -- downstream lookups never hit a missing row.
    grid AS (
        SELECT s.airport, bb.bucket AS inbound_bucket, bb.bucket_low, bb.bucket_high
        FROM airport_stats s
        CROSS JOIN {_bucket_bounds_values_sql()}
    )
    SELECT
        g.airport,
        g.inbound_bucket,
        g.bucket_low,
        g.bucket_high,
        COALESCE(pa.airport_sample_size, 0) AS airport_sample_size,
        COALESCE(pa.airport_sample_size, 0) < {MIN_BUCKET_SAMPLE_THRESHOLD} AS is_fallback,
        -- PRE-BLENDED: the airport's own capped sample if it clears the
        -- threshold, else the matching network-wide row's sample — callers
        -- never need their own fallback logic.
        CASE WHEN COALESCE(pa.airport_sample_size, 0) >= {MIN_BUCKET_SAMPLE_THRESHOLD}
             THEN pa.own_samples
             ELSE n.dep_delay_samples END AS dep_delay_samples
    FROM grid g
    LEFT JOIN per_airport pa ON pa.airport = g.airport AND pa.inbound_bucket = g.inbound_bucket
    LEFT JOIN network_node_delay_distributions n ON n.inbound_bucket = g.inbound_bucket
    """)


def build_network_edge_delay_distributions(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
    CREATE OR REPLACE TABLE network_edge_delay_distributions AS
    WITH base AS (
        SELECT
            {_bucket_case_sql('DepDelayMinutes')} AS dep_bucket,
            ArrDelayMinutes
        FROM flights_propagation
    ),
    ranked AS (
        SELECT
            dep_bucket,
            CAST(ArrDelayMinutes AS INTEGER) AS arr_delay_sample,
            ROW_NUMBER() OVER (PARTITION BY dep_bucket ORDER BY random()) AS rn
        FROM base
    )
    SELECT
        bb.bucket AS dep_bucket,
        bb.bucket_low,
        bb.bucket_high,
        COUNT(r.arr_delay_sample) AS network_sample_size,
        array_agg(r.arr_delay_sample) FILTER (WHERE r.rn <= {MAX_STORED_SAMPLES}) AS arr_delay_samples
    FROM {_bucket_bounds_values_sql()}
    LEFT JOIN ranked r ON r.dep_bucket = bb.bucket
    GROUP BY bb.bucket, bb.bucket_low, bb.bucket_high
    """)


def build_edge_delay_distributions(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
    CREATE OR REPLACE TABLE edge_delay_distributions AS
    WITH base AS (
        SELECT
            Origin AS origin,
            Dest AS dest,
            {_bucket_case_sql('DepDelayMinutes')} AS dep_bucket,
            ArrDelayMinutes
        FROM flights_propagation
    ),
    ranked AS (
        SELECT
            origin, dest, dep_bucket,
            CAST(ArrDelayMinutes AS INTEGER) AS arr_delay_sample,
            ROW_NUMBER() OVER (PARTITION BY origin, dest, dep_bucket ORDER BY random()) AS rn
        FROM base
    ),
    per_route AS (
        SELECT
            origin, dest, dep_bucket,
            COUNT(arr_delay_sample) AS route_sample_size,
            array_agg(arr_delay_sample) FILTER (WHERE rn <= {MAX_STORED_SAMPLES}) AS own_samples
        FROM ranked
        GROUP BY origin, dest, dep_bucket
    ),
    -- Full grid: every observed route (from route_stats) x every bucket.
    grid AS (
        SELECT r.Origin AS origin, r.Dest AS dest, bb.bucket AS dep_bucket, bb.bucket_low, bb.bucket_high
        FROM route_stats r
        CROSS JOIN {_bucket_bounds_values_sql()}
    )
    SELECT
        g.origin,
        g.dest,
        g.dep_bucket,
        g.bucket_low,
        g.bucket_high,
        COALESCE(pr.route_sample_size, 0) AS route_sample_size,
        COALESCE(pr.route_sample_size, 0) < {MIN_BUCKET_SAMPLE_THRESHOLD} AS is_fallback,
        -- PRE-BLENDED: the route's own capped sample if it clears the
        -- threshold, else the matching network-wide row's sample.
        CASE WHEN COALESCE(pr.route_sample_size, 0) >= {MIN_BUCKET_SAMPLE_THRESHOLD}
             THEN pr.own_samples
             ELSE n.arr_delay_samples END AS arr_delay_samples
    FROM grid g
    LEFT JOIN per_route pr ON pr.origin = g.origin AND pr.dest = g.dest AND pr.dep_bucket = g.dep_bucket
    LEFT JOIN network_edge_delay_distributions n ON n.dep_bucket = g.dep_bucket
    """)


def _print_fallback_rates(con: duckdb.DuckDBPyConnection):
    print("Fallback rates by bucket (node_delay_distributions):")
    for label, is_fb, total in con.execute("""
        SELECT inbound_bucket, SUM(CASE WHEN is_fallback THEN 1 ELSE 0 END), COUNT(*)
        FROM node_delay_distributions GROUP BY inbound_bucket ORDER BY inbound_bucket
    """).fetchall():
        print(f"  {label:<18} {is_fb}/{total} ({is_fb / total:.1%})")

    print("Fallback rates by bucket (edge_delay_distributions):")
    for label, is_fb, total in con.execute("""
        SELECT dep_bucket, SUM(CASE WHEN is_fallback THEN 1 ELSE 0 END), COUNT(*)
        FROM edge_delay_distributions GROUP BY dep_bucket ORDER BY dep_bucket
    """).fetchall():
        print(f"  {label:<18} {is_fb}/{total} ({is_fb / total:.1%})")


def main():
    con = duckdb.connect(str(DB_PATH))
    tables = con.execute("SHOW TABLES").fetchall()
    if ("flights_propagation",) not in tables:
        raise RuntimeError("Need flights_propagation. Run build_propogation.py first.")

    build_network_node_delay_distributions(con)
    build_node_delay_distributions(con)
    build_network_edge_delay_distributions(con)
    build_edge_delay_distributions(con)

    n_node = con.execute("SELECT COUNT(*) FROM node_delay_distributions").fetchone()[0]
    n_edge = con.execute("SELECT COUNT(*) FROM edge_delay_distributions").fetchone()[0]
    print(f"Built node_delay_distributions ({n_node:,} rows), "
          f"edge_delay_distributions ({n_edge:,} rows), plus network-wide "
          f"fallback tables, in {DB_PATH.resolve()}")
    _print_fallback_rates(con)
    con.close()


if __name__ == "__main__":
    main()
