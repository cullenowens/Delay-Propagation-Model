"""
build_propagation.py

Builds the analysis layer on top of `flights_enriched`:

  flights_propagation   one row per flight leg, with node/edge effects and
                        rotation context attached via window functions.
  airport_stats         per-airport aggregate: avg node effect, amplify/absorb counts.
  route_stats           per origin-dest pair: avg edge effect, amplify/absorb counts.

Run after enrich_timestamps.py. Safe to re-run — everything is CREATE OR REPLACE.

KEY DEFINITIONS
---------------
node_effect = this leg's departure delay − previous leg's arrival delay
    How much the GROUND/TURN process at the connecting airport changed the
    delay. Positive = the turn amplified the delay (fell further behind).
    Negative = the turn absorbed delay (a fast turn clawing time back).
    Only defined when there's a previous leg in the same rotation.

edge_effect_delay = this leg's arrival delay − this leg's departure delay
    Delay-differential: did lateness grow (+) or shrink (−) while the flight
    was airborne/taxiing. Negative is common and real (crew making up time).

edge_effect_schedule = actual elapsed − scheduled elapsed (CRSElapsedTime)
    Schedule-relative: did the flight take longer (+) or shorter (−) than the
    airline's own planned block time. Flags routes that are chronically padded
    (usually negative) or chronically slow (usually positive). Independent of
    how late the flight was — a flight can be schedule-slow yet still make up
    delay, so this and edge_effect_delay can disagree, on purpose.

scheduled_buffer = scheduled turn time − empirical minimum turn time
    Scheduled turn = next leg's scheduled dep − this leg's scheduled arr.
    Empirical minimum turn = a low percentile (MIN_TURN_PERCENTILE) of OBSERVED
    ground times at that airport (proxy for "the fastest a plane realistically
    turns here", since we have no aircraft-type / published-min-turn data).
    buffer = how much slack the schedule builds in beyond that floor.

propagated_delay = max(0, prev_arr_delay − scheduled_buffer)
    The buffer-absorption view: how much inbound lateness is EXPECTED to leak
    through to the next departure after the schedule's slack absorbs what it can.

ROTATION CHAINING
-----------------
Legs are chained by (PARTITION BY Tail_Number ORDER BY DepTimestampUTC) — pure
chronological order, so a rotation is tracked correctly regardless of calendar
day or time of day. A cancelled/diverted leg is EXCLUDED before chaining, so
the leg after a cancellation links to nothing (fresh start, null previous) for
this first pass. rotation_leg_number is a human-readable "nth flight of the
operating day" counter (operating day = 04:30–04:29 local origin time); it is
display-only and never used for chaining.

DEAD-BAND
---------
A node/edge effect must exceed DEADBAND_MIN minutes (default 3) to count as an
amplify or absorb event, so 1–2 minute integer jitter isn't miscounted.
"""

from pathlib import Path
import duckdb

DB_PATH = Path("flights.duckdb")
DEADBAND_MIN = 3          # minutes; effects within ±this don't count as amplify/absorb
MIN_TURN_PERCENTILE = 0.05  # 5th percentile of observed ground time = "min turn" proxy

# A "turn" longer than this (minutes) is treated as an overnight sit / long park,
# not a same-day operational turn. Such turns are EXCLUDED from the node-effect
# aggregates (airport_stats) because ~10h of slack trivially absorbs any inbound
# lateness, which would otherwise make overnight airports look like great delay
# absorbers. The raw rows stay in flights_propagation; only the aggregate filters.
# 180 min (3h) comfortably covers even wide-body international turns while
# excluding true overnight parks.
MAX_OPERATIONAL_TURN_MIN = 180

# Empirical min-turn is a percentile of observed turns at an airport. At airports
# with few sampled turns that percentile is unstable, so any airport with fewer
# than this many operational-turn samples falls back to a network-wide default
# minimum turn instead of its own thin, noisy estimate.
MIN_TURN_SAMPLE_THRESHOLD = 10


def build_propagation(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
    CREATE OR REPLACE TABLE flights_propagation AS
    WITH
    -- Only completed legs participate in rotation chaining. Cancelled and
    -- diverted legs are dropped here so the next leg becomes a fresh start.
    completed AS (
        SELECT *
        FROM flights_enriched
        WHERE Cancelled = 0 AND Diverted = 0
          AND DepTimestampUTC IS NOT NULL
          AND ArrTimestampUTC IS NOT NULL
    ),
    -- Attach previous-leg context within each tail's chronological rotation.
    -- lagged_raw first pulls the immediately preceding COMPLETED leg; then we
    -- only treat it as a real "previous leg" if the plane actually continued
    -- from where it landed (prev landing airport == this departure airport).
    -- When they don't match, the chain was broken by a cancellation, diversion,
    -- or an overnight/ferry gap — so we null the previous-leg fields and this
    -- leg becomes a fresh rotation start. This is what makes a leg following a
    -- cancellation a clean fresh start rather than silently chaining across
    -- the missing leg.
    lagged_raw AS (
        SELECT
            *,
            LAG(Dest)              OVER w AS lag_dest,
            LAG(ArrDelayMinutes)   OVER w AS lag_arr_delay,
            LAG(Flight_Number_Reporting_Airline) OVER w AS lag_flight_number,
            LAG(ArrTimestampUTC)   OVER w AS lag_arr_timestamp_utc,
            LAG(CRSArrTimestampUTC) OVER w AS lag_crs_arr_timestamp_utc,
            CAST(
                (DepTimestampUTC AT TIME ZONE origin_tz) - INTERVAL '4 hours 30 minutes'
                AS DATE
            ) AS operating_day
        FROM completed
        WINDOW w AS (PARTITION BY Tail_Number ORDER BY DepTimestampUTC)
    ),
    chained AS (
        SELECT
            * EXCLUDE (lag_dest, lag_arr_delay, lag_flight_number,
                       lag_arr_timestamp_utc, lag_crs_arr_timestamp_utc),
            -- chain_ok = the plane departed from where it last landed
            CASE WHEN lag_dest = Origin THEN lag_dest END               AS prev_dest,
            CASE WHEN lag_dest = Origin THEN lag_arr_delay END          AS prev_arr_delay,
            CASE WHEN lag_dest = Origin THEN lag_flight_number END      AS prev_flight_number,
            CASE WHEN lag_dest = Origin THEN lag_arr_timestamp_utc END  AS prev_arr_timestamp_utc,
            CASE WHEN lag_dest = Origin THEN lag_crs_arr_timestamp_utc END AS prev_crs_arr_timestamp_utc,
            (lag_dest IS NULL OR lag_dest <> Origin)                    AS is_first_leg
        FROM lagged_raw
    ),
    -- Empirical minimum turn time per airport: low percentile of observed
    -- ground times (next scheduled dep − this scheduled arr) seen at that airport.
    -- Only SAME-DAY OPERATIONAL turns count — overnight sits (> MAX_OPERATIONAL_TURN_MIN)
    -- are excluded so they can't distort the distribution.
    turns AS (
        SELECT
            prev_dest AS turn_airport,
            date_diff('minute', prev_crs_arr_timestamp_utc, CRSDepTimestampUTC) AS sched_turn_min
        FROM chained
        WHERE prev_dest IS NOT NULL
          AND prev_crs_arr_timestamp_utc IS NOT NULL
          AND date_diff('minute', prev_crs_arr_timestamp_utc, CRSDepTimestampUTC)
              BETWEEN 20 AND {MAX_OPERATIONAL_TURN_MIN}
    ),
    -- Network-wide fallback minimum turn, used for airports with too few samples
    -- to trust their own percentile.
    network_min_turn AS (
        SELECT quantile_cont(sched_turn_min, {MIN_TURN_PERCENTILE}) AS default_min_turn_min
        FROM turns
    ),
    min_turn AS (
        SELECT
            turn_airport,
            COUNT(*) AS turn_sample_size,
            quantile_cont(sched_turn_min, {MIN_TURN_PERCENTILE}) AS airport_min_turn_min
        FROM turns
        GROUP BY turn_airport
    ),
    -- Blend: airports with enough samples use their own estimate; thin airports
    -- fall back to the network-wide default.
    min_turn_final AS (
        SELECT
            mt.turn_airport,
            mt.turn_sample_size,
            CASE WHEN mt.turn_sample_size >= {MIN_TURN_SAMPLE_THRESHOLD}
                 THEN mt.airport_min_turn_min
                 ELSE nmt.default_min_turn_min END AS empirical_min_turn_min,
            (mt.turn_sample_size < {MIN_TURN_SAMPLE_THRESHOLD}) AS min_turn_is_fallback
        FROM min_turn mt
        CROSS JOIN network_min_turn nmt
    )
    SELECT
        c.Reporting_Airline,
        c.Tail_Number,
        c.Flight_Number_Reporting_Airline AS flight_number,
        c.prev_flight_number,
        c.Origin,
        c.Dest,
        c.prev_dest,
        c.FlightDate,
        c.operating_day,
        c.DepTimestampUTC,
        c.ArrTimestampUTC,
        c.DepDelayMinutes,
        c.ArrDelayMinutes,
        c.prev_arr_delay,
        c.is_first_leg,

        -- Rotation leg counter (display-only; chaining uses timestamps, not this)
        ROW_NUMBER() OVER (
            PARTITION BY c.Tail_Number, c.operating_day
            ORDER BY c.DepTimestampUTC
        ) AS rotation_leg_number,

        -- NODE EFFECT: how the turn changed the delay
        CASE WHEN c.is_first_leg THEN NULL
             ELSE c.DepDelayMinutes - c.prev_arr_delay END AS node_effect,

        -- EDGE EFFECTS (two framings)
        c.ArrDelayMinutes - c.DepDelayMinutes            AS edge_effect_delay,
        c.ActualElapsedTime - c.CRSElapsedTime           AS edge_effect_schedule,

        -- Scheduled buffer + expected propagated delay
        date_diff('minute', c.prev_crs_arr_timestamp_utc, c.CRSDepTimestampUTC) AS scheduled_turn_min,
        -- Flag overnight/long sits so aggregates can exclude them without
        -- re-deriving the threshold.
        (NOT c.is_first_leg
         AND date_diff('minute', c.prev_crs_arr_timestamp_utc, c.CRSDepTimestampUTC)
             > {MAX_OPERATIONAL_TURN_MIN})                        AS is_overnight_turn,
        mt.empirical_min_turn_min,
        mt.min_turn_is_fallback,
        CASE WHEN c.is_first_leg THEN NULL
             ELSE date_diff('minute', c.prev_crs_arr_timestamp_utc, c.CRSDepTimestampUTC)
                  - mt.empirical_min_turn_min END AS scheduled_buffer,
        CASE WHEN c.is_first_leg THEN NULL
             ELSE greatest(0, c.prev_arr_delay
                  - (date_diff('minute', c.prev_crs_arr_timestamp_utc, c.CRSDepTimestampUTC)
                     - mt.empirical_min_turn_min)) END AS propagated_delay,

        -- Cumulative delay GENERATED by this rotation's own turns (sum of node
        -- effects so far). NOT a sum of arrival delays — that would double-count
        -- inherited lateness. Reads as "delay this rotation created from turns".
        SUM(
            CASE WHEN c.is_first_leg THEN 0
                 ELSE c.DepDelayMinutes - c.prev_arr_delay END
        ) OVER (
            PARTITION BY c.Tail_Number, c.operating_day
            ORDER BY c.DepTimestampUTC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_node_delay

    FROM chained c
    LEFT JOIN min_turn_final mt ON c.prev_dest = mt.turn_airport
    """)


def build_airport_stats(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
    CREATE OR REPLACE TABLE airport_stats AS
    SELECT
        prev_dest AS airport,                       -- the turn happened at this airport
        COUNT(*)                        AS turn_events,
        ROUND(AVG(node_effect), 2)      AS avg_node_effect,
        ROUND(MEDIAN(node_effect), 2)   AS median_node_effect,
        SUM(CASE WHEN node_effect >  {DEADBAND_MIN} THEN 1 ELSE 0 END) AS amplify_events,
        SUM(CASE WHEN node_effect < -{DEADBAND_MIN} THEN 1 ELSE 0 END) AS absorb_events,
        SUM(CASE WHEN abs(node_effect) <= {DEADBAND_MIN} THEN 1 ELSE 0 END) AS neutral_events
    FROM flights_propagation
    WHERE node_effect IS NOT NULL
      -- Exclude overnight/long sits: they're parking, not operational turns, and
      -- their huge slack would make overnight airports look like delay absorbers.
      AND is_overnight_turn = FALSE
    GROUP BY prev_dest
    ORDER BY avg_node_effect DESC
    """)


def build_route_stats(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
    CREATE OR REPLACE TABLE route_stats AS
    SELECT
        Origin, Dest,
        COUNT(*)                              AS flights,
        ROUND(AVG(edge_effect_delay), 2)      AS avg_edge_effect_delay,
        ROUND(AVG(edge_effect_schedule), 2)   AS avg_edge_effect_schedule,
        SUM(CASE WHEN edge_effect_delay >  {DEADBAND_MIN} THEN 1 ELSE 0 END) AS amplify_events,
        SUM(CASE WHEN edge_effect_delay < -{DEADBAND_MIN} THEN 1 ELSE 0 END) AS absorb_events,
        SUM(CASE WHEN abs(edge_effect_delay) <= {DEADBAND_MIN} THEN 1 ELSE 0 END) AS neutral_events
    FROM flights_propagation
    GROUP BY Origin, Dest
    ORDER BY avg_edge_effect_delay DESC
    """)


def main():
    con = duckdb.connect(str(DB_PATH))
    tables = con.execute("SHOW TABLES").fetchall()
    if ("flights_enriched",) not in tables:
        raise RuntimeError("Need flights_enriched. Run enrich_timestamps.py first.")

    build_propagation(con)
    build_airport_stats(con)
    build_route_stats(con)

    n = con.execute("SELECT COUNT(*) FROM flights_propagation").fetchone()[0]
    na = con.execute("SELECT COUNT(*) FROM airport_stats").fetchone()[0]
    nr = con.execute("SELECT COUNT(*) FROM route_stats").fetchone()[0]
    print(f"Built flights_propagation ({n:,} legs), "
          f"airport_stats ({na} airports), route_stats ({nr} routes) in {DB_PATH.resolve()}")
    con.close()


if __name__ == "__main__":
    main()