# Delay-Propagation-Model

## Adding a New Month of Data

1. Download the month's BTS on-time performance export and drop it into
   `import_processing/raw_data/` (CSV or XLSX, filename doesn't matter).
2. Run the pipeline in order:

```bash
   python3 import_processing/clean_bts_data.py
   python3 import_processing/flight_enrichment.py
   python3 import_processing/build_propogation.py
   python3 import_processing/build_delay_distributions.py
   python3 import_processing/build_map_export.py
```

3. Each script is safe to re-run. `clean_bts_data.py` skips any (Year, Month,
   carrier) already loaded into `flights`, and the rest fully rebuild their
   output tables (`flights_enriched`, `flights_propagation`, `airport_stats`,
   `route_stats`, `node_delay_distributions`, `edge_delay_distributions`,
   `network_node_delay_distributions`, `network_edge_delay_distributions`)
   from whatever's currently in `flights` — so re-running the whole chain
   after adding a new month's file is always safe and won't duplicate
   anything. `build_map_export.py` re-exports the small deployable
   `app/map_data.duckdb` from the tables above; commit and push it to update
   the live app.