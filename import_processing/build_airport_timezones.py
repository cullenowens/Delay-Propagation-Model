"""
build_airport_timezones.py

Builds a small reference table mapping IATA airport codes to IANA timezone
names, plus lat/lon coordinates and a display name, sourced from the
OpenFlights airport database. This only needs to be run once (or
occasionally, if you add routes to/from an airport not already in the table).

Loads the result into flights.duckdb as a table called `airport_timezones`.
(Kept this name for backward compatibility with enrich_timestamps.py and
build_propagation.py, which join on it for timezone lookups — it now also
carries lat/lon/name, used by build_map_export.py for the map.)

NOTE: this joins on the 3-letter IATA code (Origin/Dest columns), NOT on
OriginAirportID/DestAirportID. Those numeric IDs are BTS's own internal
identifiers (useful for joining to other BTS tables like T-100/DB1B) and
have no relationship to OpenFlights' IDs — don't mix the two up.
"""

from pathlib import Path
import pandas as pd
import duckdb
import urllib.request

DB_PATH = Path("flights.duckdb")
OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
CACHE_FILE = Path("airports_raw.dat")

OPENFLIGHTS_COLUMNS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "lat", "lon", "alt", "utc_offset", "dst", "tz_name", "type", "source",
]


def fetch_openflights_data() -> Path:
    if not CACHE_FILE.exists():
        print(f"Downloading airport reference data from {OPENFLIGHTS_URL} ...")
        urllib.request.urlretrieve(OPENFLIGHTS_URL, CACHE_FILE)
    return CACHE_FILE


def build_lookup() -> pd.DataFrame:
    raw_path = fetch_openflights_data()
    df = pd.read_csv(raw_path, header=None, names=OPENFLIGHTS_COLUMNS)

    # OpenFlights uses the literal string "\N" as its null marker.
    lookup = df[(df["iata"] != r"\N") & (df["tz_name"] != r"\N")]
    lookup = lookup[["iata", "tz_name", "lat", "lon", "name", "city"]].drop_duplicates(subset="iata")
    lookup["lat"] = lookup["lat"].astype(float)
    lookup["lon"] = lookup["lon"].astype(float)
    return lookup.reset_index(drop=True)


def main():
    lookup = build_lookup()
    print(f"Built lookup table with {len(lookup)} airports.")

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE airport_timezones AS SELECT * FROM lookup")

    # Sanity check: how many distinct Origin/Dest codes in your flights table
    # (if it exists yet) DON'T have a timezone match? Worth checking after
    # every re-run, since a missing match means a flight will silently get
    # null enriched timestamps.
    tables = con.execute("SHOW TABLES").fetchall()
    if ("flights",) in tables:
        missing = con.execute("""
            SELECT DISTINCT code FROM (
                SELECT Origin AS code FROM flights
                UNION
                SELECT Dest AS code FROM flights
            ) t
            WHERE code NOT IN (SELECT iata FROM airport_timezones)
        """).fetchall()
        if missing:
            print(f"[!] {len(missing)} airport code(s) in `flights` have no "
                  f"timezone match: {[m[0] for m in missing]}")
        else:
            print("All airport codes in `flights` have a timezone match. Good.")

    con.close()


if __name__ == "__main__":
    main()


def main():
    lookup = build_lookup()
    print(f"Built lookup table with {len(lookup)} airports.")

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE airport_timezones AS SELECT * FROM lookup")

    # Sanity check: how many distinct Origin/Dest codes in your flights table
    # (if it exists yet) DON'T have a timezone match? Worth checking after
    # every re-run, since a missing match means a flight will silently get
    # null enriched timestamps.
    tables = con.execute("SHOW TABLES").fetchall()
    if ("flights",) in tables:
        missing = con.execute("""
            SELECT DISTINCT code FROM (
                SELECT Origin AS code FROM flights
                UNION
                SELECT Dest AS code FROM flights
            ) t
            WHERE code NOT IN (SELECT iata FROM airport_timezones)
        """).fetchall()
        if missing:
            print(f"[!] {len(missing)} airport code(s) in `flights` have no "
                  f"timezone match: {[m[0] for m in missing]}")
        else:
            print("All airport codes in `flights` have a timezone match. Good.")

    con.close()


if __name__ == "__main__":
    main()