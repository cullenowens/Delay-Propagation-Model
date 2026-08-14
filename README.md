# Delay-Propagation-Model

## Adding a New Month of Data

1. Download the month's BTS on-time performance export and drop it into
   `import_processing/raw_data/` (CSV or XLSX, filename doesn't matter).
2. Run the pipeline in order:

```bash
   python3 import_processing/clean_bts_data.py
   python3 import_processing/flight_enrichment.py
   python3 import_processing/build_propogation.py
```

3. Each script is safe to re-run. `clean_bts_data.py` skips any (Year, Month,
   carrier) already loaded into `flights`, and the rest fully rebuild their
   output tables (`flights_enriched`, `flights_propagation`, `airport_stats`,
   `route_stats`) from whatever's currently in `flights` — so re-running the
   whole chain after adding a new month's file is always safe and won't
   duplicate anything.