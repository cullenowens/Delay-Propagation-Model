"""
build_map_export.py

Builds a small, self-contained DuckDB file for the deployed Streamlit map —
NOT the full flights.duckdb. This is the file that actually ships to
production; everything upstream of it (raw ingestion, enrichment,
propagation) stays local/offline and never needs to leave your machine.

Produces map_data.duckdb with two pre-joined tables:
  airport_points   airport_stats + lat/lon/name, ready to plot directly.
  route_arcs       route_stats + both endpoints' lat/lon, ready to plot directly.

Plus four delay-distribution tables copied straight through (no join
needed, they're already keyed correctly) for the Monte Carlo path
simulator (app/path_simulation.py):
  node_delay_distributions, edge_delay_distributions,
  network_node_delay_distributions, network_edge_delay_distributions.
No raw per-flight/tail rows ever leave flights.duckdb — only already-
aggregated, already-capped sample arrays.

Why pre-join here instead of in the Streamlit app: it keeps app.py simple
(no joins at runtime) and keeps the exported file free of anything the app
doesn't actually need — no raw flight-level rows, no tail numbers, no
delay-cause columns. Only aggregated, airport/route-level statistics.

Re-run this any time flights.duckdb's airport_stats/route_stats/delay
distribution tables change (i.e., after re-running build_propogation.py and
build_delay_distributions.py on new data), then redeploy / push
map_data.duckdb to update the live app.
"""

from pathlib import Path
import duckdb

SOURCE_DB = Path("flights.duckdb")
EXPORT_DB = Path("app/map_data.duckdb")  # lands directly in the deployable app/ folder


def main():
    src = duckdb.connect(str(SOURCE_DB), read_only=True)

    tables = src.execute("SHOW TABLES").fetchall()
    required = {"airport_stats", "route_stats", "airport_timezones",
                "node_delay_distributions", "edge_delay_distributions",
                "network_node_delay_distributions",
                "network_edge_delay_distributions"}
    missing = required - {t[0] for t in tables}
    if missing:
        raise RuntimeError(
            f"Missing table(s) {missing} in {SOURCE_DB}. Run build_propogation.py, "
            f"build_delay_distributions.py (and build_airport_timezones.py, for "
            f"lat/lon) first."
        )

    airport_points = src.execute("""
        SELECT
            s.airport,
            tz.name  AS airport_name,
            tz.city  AS city,
            tz.lat, tz.lon,
            s.turn_events, s.avg_node_effect, s.median_node_effect,
            s.amplify_events, s.absorb_events, s.neutral_events
        FROM airport_stats s
        JOIN airport_timezones tz ON s.airport = tz.iata
    """).df()

    route_arcs = src.execute("""
        SELECT
            r.Origin AS origin, r.Dest AS dest,
            o.lat AS origin_lat, o.lon AS origin_lon,
            d.lat AS dest_lat,   d.lon AS dest_lon,
            r.flights, r.avg_edge_effect_delay, r.avg_edge_effect_schedule,
            r.amplify_events, r.absorb_events, r.neutral_events
        FROM route_stats r
        JOIN airport_timezones o ON r.Origin = o.iata
        JOIN airport_timezones d ON r.Dest = d.iata
    """).df()

    n_airports_dropped = src.execute("SELECT COUNT(*) FROM airport_stats").fetchone()[0] - len(airport_points)
    n_routes_dropped = src.execute("SELECT COUNT(*) FROM route_stats").fetchone()[0] - len(route_arcs)
    if n_airports_dropped or n_routes_dropped:
        print(f"[!] Dropped {n_airports_dropped} airport(s) and {n_routes_dropped} "
              f"route(s) with no lat/lon match in airport_timezones — check for "
              f"unusual/foreign airport codes.")

    node_dist_df = src.execute("SELECT * FROM node_delay_distributions").df()
    edge_dist_df = src.execute("SELECT * FROM edge_delay_distributions").df()
    network_node_dist_df = src.execute("SELECT * FROM network_node_delay_distributions").df()
    network_edge_dist_df = src.execute("SELECT * FROM network_edge_delay_distributions").df()

    src.close()

    if EXPORT_DB.exists():
        EXPORT_DB.unlink()
    out = duckdb.connect(str(EXPORT_DB))
    out.execute("CREATE TABLE airport_points AS SELECT * FROM airport_points")
    out.execute("CREATE TABLE route_arcs AS SELECT * FROM route_arcs")
    out.execute("CREATE TABLE node_delay_distributions AS SELECT * FROM node_dist_df")
    out.execute("CREATE TABLE edge_delay_distributions AS SELECT * FROM edge_dist_df")
    out.execute("CREATE TABLE network_node_delay_distributions AS SELECT * FROM network_node_dist_df")
    out.execute("CREATE TABLE network_edge_delay_distributions AS SELECT * FROM network_edge_dist_df")
    out.close()

    size_kb = EXPORT_DB.stat().st_size / 1024
    print(f"Wrote {EXPORT_DB.resolve()} ({size_kb:.1f} KB) — "
          f"{len(airport_points)} airports, {len(route_arcs)} routes, "
          f"{len(node_dist_df)} node dist rows, {len(edge_dist_df)} edge dist rows.")


if __name__ == "__main__":
    main()