"""

Reads the raw `flights` table (local HHMM times, ambiguous timezone) and
writes a new `flights_enriched` table containing real, unambiguous UTC
timestamps for scheduled/actual departure and arrival.

WHY A SEPARATE TABLE, BUILT BY A SEPARATE SCRIPT
--------------------------------------------------
This keeps the timezone logic isolated from the raw cleaning/loading step
(clean_bts_data.py). If you find a bad timezone mapping or a bug in this
logic later, you fix it here and re-run just this script — you never need
to re-download or re-clean the raw BTS files. `flights` stays your
untouched source-of-truth layer; `flights_enriched` is a derived,
rebuildable layer on top of it.

Re-run this script any time you load new months into `flights` or update
airport_timezones.csv — it always rebuilds `flights_enriched` from
scratch (CREATE OR REPLACE), so it's safe to re-run as often as you like.

THE CORE TRICK: GO THROUGH UTC, DON'T GUESS AT DATE ROLLOVER
----------------------------------------------------------------
CRSDepTime/DepTime are local clock time at the ORIGIN. CRSArrTime/ArrTime
are local clock time at the DESTINATION. Naively comparing the two HHMM
values to guess whether a flight arrived "the next day" breaks constantly
(overnight flights, flights that cross the date line, short flights that
still cross a timezone, etc).

Instead:
  1. Combine FlightDate + local departure HHMM + the ORIGIN's timezone to
     get an unambiguous UTC departure timestamp.
  2. Add the elapsed time (in minutes — timezone-agnostic) to get the UTC
     arrival timestamp. No date-rollover guessing needed at all.
  3. (Optional, done here as a sanity check) Convert that UTC arrival back
     to the destination's local time and compare it to the raw ArrTime —
     they should match. Mismatches usually mean a bad timezone lookup or
     a DST-transition edge case worth a second look.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

DB_PATH = Path("flights.duckdb")


def hhmm_to_hour_minute_dayshift(hhmm: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split an HHMM int column into (hour, minute, day_offset).

    BTS uses "2400" to mean midnight at the END of the scheduled day
    (i.e. hour 0 of the NEXT day), not hour 24. This handles that.
    """
    hhmm = hhmm.astype("Int64")
    hour = (hhmm // 100).astype("Int64")
    minute = (hhmm % 100).astype("Int64")
    day_offset = (hour == 24).astype("Int64")
    hour = hour.where(hour != 24, 0)
    return hour, minute, day_offset


def localize_to_utc(flight_date: pd.Series, hhmm: pd.Series, tz_name: pd.Series) -> pd.Series:
    """Combine a date + local HHMM time + IANA timezone name into a UTC
    timestamp, row by row but grouped by timezone for speed (pandas can't
    tz_localize a Series with mixed timezones in one call)."""
    hour, minute, day_offset = hhmm_to_hour_minute_dayshift(hhmm)

    naive = (
        flight_date
        + pd.to_timedelta(day_offset.fillna(0).astype(int), unit="D")
        + pd.to_timedelta(hour.fillna(0).astype(int), unit="h")
        + pd.to_timedelta(minute.fillna(0).astype(int), unit="m")
    )
    # Preserve nulls (cancelled flights etc.) — don't fabricate a timestamp
    # for rows where the source HHMM was missing.
    naive = naive.where(hhmm.notna() & tz_name.notna())

    result = pd.Series(pd.NaT, index=flight_date.index, dtype="datetime64[ns, UTC]")
    for tz, group_idx in naive.groupby(tz_name).groups.items():
        valid_idx = group_idx.intersection(naive[naive.notna()].index)
        if len(valid_idx) == 0:
            continue
        localized = naive.loc[valid_idx].dt.tz_localize(
            tz, ambiguous="NaT", nonexistent="NaT"
        )
        result.loc[valid_idx] = localized.dt.tz_convert("UTC")
    return result


def enrich() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))

    tables = con.execute("SHOW TABLES").fetchall()
    if ("flights",) not in tables or ("airport_timezones",) not in tables:
        raise RuntimeError(
            "Need both `flights` and `airport_timezones` tables in "
            f"{DB_PATH}. Run clean_bts_data.py and build_airport_timezones.py first."
        )

    df = con.execute("""
        SELECT f.*, o.tz_name AS origin_tz, d.tz_name AS dest_tz
        FROM flights f
        LEFT JOIN airport_timezones o ON f.Origin = o.iata
        LEFT JOIN airport_timezones d ON f.Dest = d.iata
    """).df()

    con.close()

    n_missing_tz = df["origin_tz"].isna().sum() + df["dest_tz"].isna().sum()
    if n_missing_tz:
        print(f"[!] {n_missing_tz} origin/dest values had no timezone match "
              f"— those rows will have null enriched timestamps.")

    # --- Scheduled times ---
    df["CRSDepTimestampUTC"] = localize_to_utc(df["FlightDate"], df["CRSDepTime"], df["origin_tz"])
    df["CRSArrTimestampUTC"] = df["CRSDepTimestampUTC"] + pd.to_timedelta(
        df["CRSElapsedTime"].astype("Float64"), unit="m"
    )

    # --- Actual times ---
    df["DepTimestampUTC"] = localize_to_utc(df["FlightDate"], df["DepTime"], df["origin_tz"])
    df["ArrTimestampUTC"] = df["DepTimestampUTC"] + pd.to_timedelta(
        df["ActualElapsedTime"].astype("Float64"), unit="m"
    )

    # --- Sanity check: convert computed UTC arrival back to destination
    # local time and compare against the raw ArrTime BTS gave us.
    #
    # NOTE: this has to compute hour/minute INSIDE each timezone group,
    # not by reassembling one combined column afterward — pandas can't
    # hold mixed timezones in a single column, so if you store the
    # per-group localized values into a column that's typed as UTC, it
    # silently converts them straight back to UTC and you end up
    # comparing UTC-hour against local ArrTime, which will never match.
    computed_hhmm = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for tz, group_idx in df.groupby("dest_tz").groups.items():
        valid_idx = group_idx.intersection(df[df["ArrTimestampUTC"].notna()].index)
        if len(valid_idx) == 0:
            continue
        local = df.loc[valid_idx, "ArrTimestampUTC"].dt.tz_convert(tz)
        computed_hhmm.loc[valid_idx] = (local.dt.hour * 100 + local.dt.minute).astype("Float64")

    raw_hhmm = df["ArrTime"].astype("Float64")
    mismatch = (
        computed_hhmm.notna() & raw_hhmm.notna()
        & (computed_hhmm.astype("Float64") != raw_hhmm)
    )
    n_mismatch = mismatch.sum()
    n_checked = (computed_hhmm.notna() & raw_hhmm.notna()).sum()
    print(f"Sanity check: {n_checked - n_mismatch:,} / {n_checked:,} actual "
          f"arrivals match raw ArrTime when reconstructed from UTC "
          f"({n_mismatch:,} mismatches — expect a handful from DST "
          f"transition weekends, investigate if this is a large fraction).")

    return df


def main():
    df = enrich()

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE flights_enriched AS SELECT * FROM df")
    n_rows = con.execute("SELECT COUNT(*) FROM flights_enriched").fetchone()[0]
    con.close()

    print(f"\nDone. `flights_enriched` now has {n_rows:,} rows in {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()