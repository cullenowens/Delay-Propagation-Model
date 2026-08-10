"""
clean_bts_data.py

Cleans monthly BTS "Airline On-Time Performance" files (CSV or XLSX, as
downloaded one-month-at-a-time from transtats.bts.gov) and loads them into
a DuckDB database for analysis.

USAGE
-----
1. Drop your monthly BTS export files into the RAW_DATA_DIR folder below
   (filenames don't matter, extension can be .csv or .xlsx).
2. Run:  python clean_bts_data.py
3. Query the result with DuckDB, e.g.:
       import duckdb
       con = duckdb.connect("flights.duckdb")
       con.sql("SELECT * FROM flights LIMIT 10").show()

WHAT THIS SCRIPT DOES
----------------------
- Keeps only the carriers listed in TARGET_CARRIERS (add more anytime —
  re-run against the same raw files if you expand the list later, no
  need to re-download).
- Drops the bulk diversion-specific columns (Div1Airport...Div5TailNum,
  FirstDepTime, TotalAddGTime, etc.) — these are >99% empty and only
  matter if you're specifically studying diversions.
- Keeps OriginAirportID / DestAirportID (not just the 3-letter code) since
  those numeric IDs are the join keys used by other BTS tables like
  T-100 and DB1B, which you'll likely want later.
- Fixes the two biggest gotchas in this dataset:
    1. CarrierDelay/WeatherDelay/NASDelay/SecurityDelay/LateAircraftDelay
       are blank (NaN) whenever a flight's arrival delay was under 15
       minutes — NOT because the cause is unknown, but because BTS only
       attributes cause for delays >= 15 min. Leaving these as NaN will
       silently break any groupby/sum. This script fills them with 0.
    2. Raw BTS exports often have a trailing empty "Unnamed: N" column
       from a trailing comma in the source file. This script ignores
       any column not explicitly on the keep-list, so stray columns
       never make it into the database.
- Skips a raw file if that (Year, Month) combination is already loaded,
  so you can safely re-run the script if you add new monthly files.

HEADS-UP FOR THE NEXT STAGE (not handled here, on purpose)
------------------------------------------------------------
CRSDepTime/DepTime are LOCAL time at the ORIGIN airport, but
CRSArrTime/ArrTime are LOCAL time at the DESTINATION airport. When you
build real timestamps for the rotation-reconstruction model, you can't
just combine FlightDate + HHMM across a flight's origin and destination
— you'll need an airport -> timezone lookup (e.g. OpenFlights' airports.dat)
to normalize everything to UTC first. This script leaves the raw HHMM
values as-is (cleaned to a nullable integer) rather than guessing at
timezones, so you don't bake in a silent error. Happy to help build that
lookup table when you get to that step.
"""

from pathlib import Path
import pandas as pd
import duckdb

# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------

# Delta now; United is already in the list as foresight for phase 2.
# Adding a carrier later just means re-running this script against the
# same raw files — you don't need to re-download anything.
TARGET_CARRIERS = ["DL", "UA"]

RAW_DATA_DIR = Path("import_processing/raw_data")      # put your monthly BTS files here
DB_PATH = Path("flights.duckdb")
TABLE_NAME = "flights"

# Columns to keep. Anything not listed here is dropped, including the
# diversion columns and any stray "Unnamed" columns from the raw export.
KEEP_COLUMNS = [
    # Calendar / identifiers
    "Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline",
    # Origin / Destination — AirportID kept for future T-100 / DB1B joins
    "OriginAirportID", "Origin", "OriginCityName", "OriginState",
    "DestAirportID", "Dest", "DestCityName", "DestState",
    # Scheduled vs. actual times (raw HHMM ints — see timezone note above)
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrDel15",
    # Cancellations / diversions — need these to know a rotation broke
    "Cancelled", "CancellationCode", "Diverted",
    # Duration / distance
    "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
    # Delay-cause breakdown — the core of the propagation model
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay",
]

# These are only populated for delayed (>=15 min) flights; NaN elsewhere
# means "not applicable," which we treat as 0, not missing data.
DELAY_CAUSE_COLUMNS = [
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay",
]

# Nullable integer columns (HHMM times, taxi/elapsed minutes, etc.) —
# cast with pandas' nullable "Int64" so cancelled-flight blanks stay
# blank instead of forcing the whole column to float (830 -> 830.0).
NULLABLE_INT_COLUMNS = [
    "CRSDepTime", "DepTime", "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "CRSArrTime", "ArrTime", "CRSElapsedTime", "ActualElapsedTime",
    "AirTime", "Distance", "Flight_Number_Reporting_Airline",
    "OriginAirportID", "DestAirportID",
]


# ----------------------------------------------------------------------
# CLEANING
# ----------------------------------------------------------------------

def load_raw_file(filepath: Path) -> pd.DataFrame:
    """Read a raw BTS export, whether it's CSV or Excel."""
    if filepath.suffix.lower() == ".csv":
        return pd.read_csv(filepath, low_memory=False)
    elif filepath.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unrecognized file type: {filepath}")


def clean_month(filepath: Path) -> pd.DataFrame:
    """Load one monthly BTS file and return a cleaned, filtered DataFrame."""
    df = load_raw_file(filepath)

    # Filter to the carriers we care about, before doing anything else —
    # this is what keeps memory use down on the full-network files.
    df = df[df["Reporting_Airline"].isin(TARGET_CARRIERS)].copy()

    # Keep only columns we actually want (drops diversion bulk + any
    # stray "Unnamed: N" columns from trailing commas in the source file).
    # Missing columns are ignored rather than erroring, in case a future
    # month's export drops or renames something.
    available_cols = [c for c in KEEP_COLUMNS if c in df.columns]
    missing_cols = set(KEEP_COLUMNS) - set(available_cols)
    if missing_cols:
        print(f"  [!] {filepath.name}: missing expected columns {missing_cols}")
    df = df[available_cols]

    # Delay-cause columns: NaN means "not applicable" (delay < 15 min or
    # flight cancelled/diverted), not "unknown" — fill with 0 so sums and
    # groupbys behave correctly.
    for col in DELAY_CAUSE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Nullable integer casting so cancelled flights (blank times) don't
    # force these columns to float.
    for col in NULLABLE_INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Dates and simple string cleanup.
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    if "Tail_Number" in df.columns:
        df["Tail_Number"] = df["Tail_Number"].astype("string").str.strip()
    if "CancellationCode" in df.columns:
        df["CancellationCode"] = df["CancellationCode"].astype("string").str.strip()

    for col in ("Cancelled", "Diverted", "DepDel15", "ArrDel15"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    return df


# ----------------------------------------------------------------------
# DATABASE LOADING
# ----------------------------------------------------------------------

def already_loaded(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> bool:
    """Check whether this file's (Year, Month, carrier) combo is already
    in the database, so re-running the script doesn't duplicate rows."""
    tables = con.execute("SHOW TABLES").fetchall()
    if (TABLE_NAME,) not in tables:
        return False
    year, month = int(df["Year"].iloc[0]), int(df["Month"].iloc[0])
    carriers = tuple(df["Reporting_Airline"].unique())
    existing = con.execute(
        f"""
        SELECT COUNT(*) FROM {TABLE_NAME}
        WHERE Year = ? AND Month = ? AND Reporting_Airline IN ?
        """,
        [year, month, carriers],
    ).fetchone()[0]
    return existing > 0


def load_to_duckdb(df: pd.DataFrame, con: duckdb.DuckDBPyConnection):
    tables = con.execute("SHOW TABLES").fetchall()
    if (TABLE_NAME,) not in tables:
        con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df")
    else:
        con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM df")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    RAW_DATA_DIR.mkdir(exist_ok=True)
    files = sorted(RAW_DATA_DIR.glob("*.csv")) + sorted(RAW_DATA_DIR.glob("*.xlsx"))

    if not files:
        print(f"No files found in {RAW_DATA_DIR.resolve()}. Drop your monthly "
              f"BTS exports there and re-run.")
        return

    con = duckdb.connect(str(DB_PATH))
    total_rows = 0

    for filepath in files:
        df = clean_month(filepath)

        if df.empty:
            print(f"  - {filepath.name}: no rows matched TARGET_CARRIERS, skipping")
            continue

        if already_loaded(con, df):
            print(f"  - {filepath.name}: already in {DB_PATH}, skipping")
            continue

        load_to_duckdb(df, con)
        total_rows += len(df)
        print(f"  + {filepath.name}: loaded {len(df):,} rows")

    grand_total = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"\nDone. Added {total_rows:,} rows this run. "
          f"Table '{TABLE_NAME}' now has {grand_total:,} rows total "
          f"in {DB_PATH.resolve()}")

    con.close()


if __name__ == "__main__":
    main()