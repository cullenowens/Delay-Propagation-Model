"""
app.py

Streamlit + pydeck dashboard showing which airports amplify or absorb delay
(node effect) and which routes gain or lose delay in the air (edge effect).

Reads ONLY map_data.duckdb — the small pre-aggregated export produced by
build_map_export.py. Never touches the full flights.duckdb / raw pipeline,
by design: this file is what actually deploys.

COLOR SCALE: fixed (not relative to whatever's on screen), so colors mean
the same thing regardless of filters: green at -CAP_MINUTES (strong
absorber), blue at 0 (neutral), red at +CAP_MINUTES (strong amplifier).
Same scale used for airports (node effect) and routes (edge effect).

INTERACTIVITY: the bottom section is two tabs (Airports / Routes) instead
of two always-open tables. Each tab supports multi-row selection, and both
tabs' selections combine (union) into one "focus" in st.session_state.
The map and the tables read that focus differently:
  - the MAP shows only the exact selected airports/routes (exact_focus) —
    no expansion, so it never shows more than what you picked
  - the TABLES show the exact selection plus everything connected to it
    (apply_focus) — selecting an airport also surfaces routes touching
    it, and vice versa, so you can browse outward from a pick
Threshold sliders / show-routes / show-airports toggles in the sidebar, AND
row selection in the tables below, are all STAGED, not live — the map only
updates when the "Go" button is clicked. "Go" and "Clear selection" sit
together below the map, near the tables.

Run locally:   streamlit run app.py
Deploy:        push this repo (with map_data.duckdb committed) to GitHub,
               then point Streamlit Community Cloud at it. See README for
               the full deployment checklist.
"""

import re
from pathlib import Path

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from path_simulation import simulate_path, summarize, N_DRAWS_DEFAULT

DB_PATH = Path(__file__).parent / "map_data.duckdb"

# Full color saturation at +/- this many minutes. Same cap used for both
# node effect (airports) and edge effect (routes), per the request that
# they share one consistent scale.
CAP_MINUTES = 15

GREEN = (44, 160, 68)   # strong absorber (-CAP_MINUTES or beyond)
BLUE = (31, 119, 180)   # neutral (0 minutes)
RED = (214, 39, 40)     # strong amplifier (+CAP_MINUTES or beyond)
GRAY = (170, 170, 170)  # missing/undefined value

st.set_page_config(page_title="Delay Propagation Map", layout="wide")


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

@st.cache_resource
def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_data
def load_data():
    con = get_connection()
    airports = con.execute("SELECT * FROM airport_points").df()
    routes = con.execute("SELECT * FROM route_arcs").df()
    return airports, routes


@st.cache_data
def load_distributions():
    con = get_connection()
    node_df = con.execute("SELECT * FROM node_delay_distributions").df()
    edge_df = con.execute("SELECT * FROM edge_delay_distributions").df()
    net_node_df = con.execute("SELECT * FROM network_node_delay_distributions").df()
    net_edge_df = con.execute("SELECT * FROM network_edge_delay_distributions").df()
    node_lookup = {(r.airport, r.inbound_bucket): np.asarray(r.dep_delay_samples) for r in node_df.itertuples()}
    edge_lookup = {(r.origin, r.dest, r.dep_bucket): np.asarray(r.arr_delay_samples) for r in edge_df.itertuples()}
    network_node_lookup = {r.inbound_bucket: np.asarray(r.dep_delay_samples) for r in net_node_df.itertuples()}
    network_edge_lookup = {r.dep_bucket: np.asarray(r.arr_delay_samples) for r in net_edge_df.itertuples()}
    return node_lookup, edge_lookup, network_node_lookup, network_edge_lookup


# ----------------------------------------------------------------------
# COLOR MAPPING — fixed three-stop diverging scale: green -> blue -> red.
# Unlike a scale relative to the current data's max, this means a +12 min
# airport always renders the same color whether it's on screen alone or
# next to a +45 min outlier — colors are comparable across every filter
# state and across sessions.
# ----------------------------------------------------------------------

def _lerp(a, b, t):
    return [int(a[i] + (b[i] - a[i]) * t) for i in range(3)]


def fixed_diverging_color(value: float, cap: float = CAP_MINUTES, alpha: int = 210) -> list:
    if pd.isna(value):
        return list(GRAY) + [alpha]
    v = max(-cap, min(cap, value))
    if v >= 0:
        rgb = _lerp(BLUE, RED, v / cap if cap else 0)
    else:
        rgb = _lerp(BLUE, GREEN, -v / cap if cap else 0)
    return rgb + [alpha]


def render_legend(cap: float = CAP_MINUTES):
    """A visible gradient bar with numeric tick labels, used for both the
    airport (node effect) and route (edge effect) color scales since they
    share the same mapping."""
    gradient_css = f"rgb{GREEN}, rgb{BLUE}, rgb{RED}"
    st.markdown(
        f"""
        <div style="margin-bottom: 0.5rem;">
            <div style="height: 14px; border-radius: 7px;
                        background: linear-gradient(to right, {gradient_css});
                        border: 1px solid rgba(128,128,128,0.4);"></div>
            <div style="display: flex; justify-content: space-between;
                        font-size: 0.8rem; color: gray; margin-top: 2px;">
                <span>&minus;{cap} min (absorbs)</span>
                <span>0 min (neutral)</span>
                <span>+{cap} min (amplifies)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Bounds-safe selection resolution. Streamlit persists a table widget's
# selected row INDEX across reruns as long as its key doesn't change — but
# the data shown can shrink between reruns (focus narrowing the table,
# or the search box narrowing it) while the widget keeps the same key.
# When that happens the remembered index can point past the end of the
# now-smaller table. This must always be bounds-checked before indexing,
# regardless of what caused the mismatch — a pure function so it's
# directly testable without spinning up Streamlit.
# ----------------------------------------------------------------------

def resolve_selection(shown: pd.DataFrame, selection_rows: list):
    valid = [i for i in selection_rows if i < len(shown)]  # drop stale indices from before the table shrank
    if not valid:
        return None
    return shown.iloc[valid]


# ----------------------------------------------------------------------
# CROSS-FILTERING — a pure function (no Streamlit calls) so it's testable
# on its own: given the threshold-filtered base data and an optional set
# of focused airports/routes, return the final airports/routes to
# actually display. The two focus lists combine (union), not either/or —
# selecting airports AND routes at once shows everything touched by
# either selection.
# ----------------------------------------------------------------------

def apply_focus(airports_base: pd.DataFrame, routes_base: pd.DataFrame,
                 focus_airports: list, focus_routes: list):
    if not focus_airports and not focus_routes:
        return airports_base.copy(), routes_base.copy()

    airport_codes = set(focus_airports)
    route_mask = pd.Series(False, index=routes_base.index)
    if airport_codes:
        route_mask |= routes_base["origin"].isin(airport_codes) | routes_base["dest"].isin(airport_codes)
    if focus_routes:
        route_pairs = set(focus_routes)
        pairs = pd.Series(list(zip(routes_base["origin"], routes_base["dest"])), index=routes_base.index)
        route_mask |= pairs.isin(route_pairs)
    routes_final = routes_base[route_mask]

    connected = set(routes_final["origin"]) | set(routes_final["dest"]) | airport_codes
    airports_final = airports_base[airports_base["airport"].isin(connected)]
    return airports_final.copy(), routes_final.copy()


# ----------------------------------------------------------------------
# EXACT FOCUS — unlike apply_focus (which expands to everything touching
# a selection, for the tables), this returns only the literal selected
# rows. Used for the map: the map should show exactly what you picked,
# not everything correlated with it. Cross-filtering context still lives
# in the Data tables below via apply_focus.
# ----------------------------------------------------------------------

def exact_focus(airports_base: pd.DataFrame, routes_base: pd.DataFrame,
                 focus_airports: list, focus_routes: list):
    if not focus_airports and not focus_routes:
        return airports_base.copy(), routes_base.copy()

    airports_final = airports_base[airports_base["airport"].isin(focus_airports)]
    route_pairs = set(focus_routes)
    pairs = pd.Series(list(zip(routes_base["origin"], routes_base["dest"])), index=routes_base.index)
    routes_final = routes_base[pairs.isin(route_pairs)]
    return airports_final.copy(), routes_final.copy()


# ----------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------

def main():
    if not DB_PATH.exists():
        st.error(
            "map_data.duckdb not found next to app.py. Run "
            "`python build_map_export.py` first, then commit the resulting "
            "file alongside app.py."
        )
        st.stop()

    airports, routes = load_data()
    node_lookup, edge_lookup, network_node_lookup, network_edge_lookup = load_distributions()
    known_airports = {a for a, _ in node_lookup}

    page_map, page_sim = st.tabs(["Delay Map", "Flight Path Simulator"])
    with page_map:
        render_map_page(airports, routes)
    with page_sim:
        render_path_simulator(node_lookup, edge_lookup, network_node_lookup,
                               network_edge_lookup, known_airports)


def render_map_page(airports: pd.DataFrame, routes: pd.DataFrame):
    st.session_state.setdefault("focus_airports", [])
    st.session_state.setdefault("focus_routes", [])
    st.session_state.setdefault("pending_focus_airports", [])
    st.session_state.setdefault("pending_focus_routes", [])
    st.session_state.setdefault("table_reset", 0)
    st.session_state.setdefault("applied_min_turns", 0)
    st.session_state.setdefault("applied_min_flights", 0)
    st.session_state.setdefault("applied_show_routes", True)
    st.session_state.setdefault("applied_show_airports", True)

    st.title("Delay Propagation Map")
    st.caption(
        "Airports colored by node effect (did the turn amplify or absorb "
        "delay); routes colored by edge effect (did the flight gain or "
        "lose delay in the air). Same color scale for both."
    )

    with st.sidebar:
        st.header("Filters")
        st.caption("Adjust these, then click **Go** (below the map) to apply.")
        min_turns_input = st.slider(
            "Minimum turn events (airports)", 0,
            int(airports["turn_events"].max()), st.session_state["applied_min_turns"],
            help="Hide airports with too few sampled turns to trust their average.",
        )
        min_flights_input = st.slider(
            "Minimum flights (routes)", 0,
            int(routes["flights"].max()), st.session_state["applied_min_flights"],
            help="Hide routes with too few flights to trust their average.",
        )
        show_routes_input = st.checkbox("Show routes", value=st.session_state["applied_show_routes"])
        show_airports_input = st.checkbox("Show airports", value=st.session_state["applied_show_airports"])

    # Everything below reads APPLIED state, not the widgets above directly —
    # this is what makes the sidebar "staged": dragging a slider alone
    # doesn't touch the map until Go copies these into applied_* and reruns.
    min_turns = st.session_state["applied_min_turns"]
    min_flights = st.session_state["applied_min_flights"]
    show_routes = st.session_state["applied_show_routes"]
    show_airports = st.session_state["applied_show_airports"]

    airports_base = airports[airports["turn_events"] >= min_turns]
    routes_base = routes[routes["flights"] >= min_flights]

    focus_airports = st.session_state["focus_airports"]
    focus_routes = st.session_state["focus_routes"]
    focus_active = bool(focus_airports or focus_routes)

    # Table data: the cross-filtered/"connected" set (selecting an airport
    # also pulls in routes touching it, and vice versa) — lets you browse
    # outward from a selection in the Data tables below.
    table_airports_f, table_routes_f = apply_focus(airports_base, routes_base, focus_airports, focus_routes)

    # Map data: ONLY the exact rows selected, no expansion — the map shows
    # what you picked, not everything correlated with it.
    map_airports_f, map_routes_f = exact_focus(airports_base, routes_base, focus_airports, focus_routes)

    # Informational only here — the actual Clear action lives below the map,
    # next to Go, per the requested layout.
    if focus_active:
        label_parts = []
        if focus_airports:
            label_parts.append(", ".join(focus_airports))
        if focus_routes:
            label_parts.append(", ".join(f"{o} → {d}" for o, d in focus_routes))
        st.info(
            f"Focused on **{'; '.join(label_parts)}** — map shows only the "
            "selected airports/routes; the tables below also show what's connected to them."
        )

    if table_airports_f.empty and table_routes_f.empty:
        st.warning("No airports or routes match the current filters.")
        st.stop()

    leg_col1, leg_col2 = st.columns(2)
    with leg_col1:
        st.markdown("**Airports** (node effect)")
        render_legend()
    with leg_col2:
        st.markdown("**Routes** (edge effect)")
        render_legend()

    map_airports_f["color"] = map_airports_f["avg_node_effect"].apply(fixed_diverging_color)
    max_turns = airports_base["turn_events"].max()
    map_airports_f["radius"] = 8000 + (map_airports_f["turn_events"] / max_turns) * 35000

    map_routes_f["color"] = map_routes_f["avg_edge_effect_delay"].apply(
        lambda v: fixed_diverging_color(v, alpha=170)
    )
    max_flights = routes_base["flights"].max()
    map_routes_f["width"] = 1 + (map_routes_f["flights"] / max_flights) * 6

    layers = []
    if show_routes and not map_routes_f.empty:
        layers.append(pdk.Layer(
            "ArcLayer",
            data=map_routes_f,
            get_source_position=["origin_lon", "origin_lat"],
            get_target_position=["dest_lon", "dest_lat"],
            get_source_color="color",
            get_target_color="color",
            get_width="width",
            pickable=True,
            auto_highlight=True,
        ))
    if show_airports and not map_airports_f.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=map_airports_f,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
        ))

    if focus_active and not map_airports_f.empty:
        view_state = pdk.ViewState(
            latitude=map_airports_f["lat"].mean(), longitude=map_airports_f["lon"].mean(),
            zoom=4.2, pitch=20,
        )
    elif focus_active and not map_routes_f.empty:
        # Route-only focus (no airports explicitly selected) — center on the
        # selected routes' endpoints instead.
        lats = pd.concat([map_routes_f["origin_lat"], map_routes_f["dest_lat"]])
        lons = pd.concat([map_routes_f["origin_lon"], map_routes_f["dest_lon"]])
        view_state = pdk.ViewState(latitude=lats.mean(), longitude=lons.mean(), zoom=4.2, pitch=20)
    else:
        view_state = pdk.ViewState(latitude=39.8, longitude=-98.6, zoom=3.2, pitch=20)

    tooltip = {
        "html": "<b>{airport}</b> {airport_name}<br/>"
                "Avg node effect: {avg_node_effect} min<br/>"
                "Turn events: {turn_events}<br/>"
                "<b>{origin} → {dest}</b><br/>"
                "Avg edge effect: {avg_edge_effect_delay} min<br/>"
                "Flights: {flights}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None,  # no Mapbox token required — pydeck's default (Carto) basemap
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True, height=600)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Airports shown", len(map_airports_f))
    col2.metric("Routes shown", len(map_routes_f))
    col3.metric("Amplifying airports", int((map_airports_f["avg_node_effect"] > 0).sum()))
    col4.metric("Absorbing airports", int((map_airports_f["avg_node_effect"] < 0).sum()))

    # Rows checked in the tables below only land in pending_focus_* (see
    # render_tables) — they aren't applied to focus_* until Go is clicked,
    # same staging rule as the sidebar sliders. This surfaces what's staged.
    pending_airports = st.session_state["pending_focus_airports"]
    pending_routes = st.session_state["pending_focus_routes"]
    pending_changed = (pending_airports != focus_airports) or (pending_routes != focus_routes)
    if pending_changed and (pending_airports or pending_routes):
        parts = []
        if pending_airports:
            parts.append(f"{len(pending_airports)} airport(s)")
        if pending_routes:
            parts.append(f"{len(pending_routes)} route(s)")
        st.caption(f"Selected {' and '.join(parts)} — click **Go** to focus the map on it.")

    # Go applies the sidebar's staged filter selections and any staged table
    # selection; Clear resets any airport/route focus from the tables below.
    # Placed together here (below the map, right above the data tables)
    # rather than in the sidebar.
    go_col, clear_col, _ = st.columns([1, 1, 4])
    if go_col.button("Go", type="primary", use_container_width=True):
        st.session_state["applied_min_turns"] = min_turns_input
        st.session_state["applied_min_flights"] = min_flights_input
        st.session_state["applied_show_routes"] = show_routes_input
        st.session_state["applied_show_airports"] = show_airports_input
        if pending_changed:
            st.session_state["focus_airports"] = list(pending_airports)
            st.session_state["focus_routes"] = list(pending_routes)
            st.session_state["table_reset"] += 1
        st.rerun()
    if clear_col.button("Clear selection", use_container_width=True, disabled=not focus_active):
        st.session_state["focus_airports"] = []
        st.session_state["focus_routes"] = []
        st.session_state["pending_focus_airports"] = []
        st.session_state["pending_focus_routes"] = []
        st.session_state["table_reset"] += 1  # forces table widgets to drop their old row highlight
        st.rerun()

    render_tables(table_airports_f, table_routes_f)


# ----------------------------------------------------------------------
# FLIGHT PATH SIMULATOR — Monte Carlo distribution forecast for a
# hypothetical multi-leg path. Reuses the staged-selection convention:
# widgets feed a local `path`/`start_condition`, but nothing simulates
# until "Simulate" is clicked, which stores results in
# st.session_state["sim_results"]; rendering below reads that stored
# result, not the live widgets, so tweaking an input never silently
# changes what's on screen until you explicitly re-run.
# ----------------------------------------------------------------------

def _parse_path_input(text: str) -> list:
    """Splits on commas, arrows ("->", "→"), and whitespace — so both
    "ATL, SAF, LAX" and "ATL -> SAF -> LAX" work identically."""
    tokens = re.split(r"[,\->→\s]+", text.strip())
    return [t.upper() for t in tokens if t]


def render_path_simulator(node_lookup: dict, edge_lookup: dict,
                           network_node_lookup: dict, network_edge_lookup: dict,
                           known_airports: set):
    st.title("Flight Path Simulator")
    st.caption(
        "Build a hypothetical multi-leg path — it doesn't need to be a route "
        "any tail has actually flown — and see a predicted delay "
        "distribution at each stop, resampled from real historical outcomes "
        "bucketed by inbound-delay size and airport/route identity."
    )

    st.session_state.setdefault("sim_results", None)

    path_text = st.text_input(
        "Path (airport codes, separated by commas or arrows)",
        value="ATL -> ORD -> LGA",
        help="e.g. ATL -> SAF -> LAX -> LGA",
    )
    path = _parse_path_input(path_text)
    unknown = [a for a in path if a not in known_airports]

    if unknown:
        st.error(f"Unknown airport code(s): {', '.join(unknown)} — not present in the dataset.")
    elif len(path) < 2:
        st.error("Enter at least two airports.")
    path_valid = len(path) >= 2 and not unknown

    start_choice = st.radio("Starting condition", ["On-time", "Assume a delay"], horizontal=True)
    if start_choice == "Assume a delay":
        start_minutes = st.number_input(
            "Minutes late departing the first airport", min_value=0, value=15, step=5,
        )
        start_condition = float(start_minutes)
    else:
        start_condition = "on_time"

    with st.expander("Advanced"):
        n_draws = st.slider("Monte Carlo draws", 200, 5000, N_DRAWS_DEFAULT, step=200)
        use_fixed_seed = st.checkbox("Use a fixed random seed (reproducible results)")
    seed = 42 if use_fixed_seed else None

    if st.button("Simulate", type="primary", disabled=not path_valid):
        results = simulate_path(
            path, start_condition,
            node_lookup, edge_lookup, network_node_lookup, network_edge_lookup,
            n_draws=n_draws, seed=seed,
        )
        st.session_state["sim_results"] = {"path": path, "start_condition": start_condition, "results": results}
        st.rerun()

    sim = st.session_state["sim_results"]
    if sim is None:
        return

    start_delay = 0.0 if sim["start_condition"] == "on_time" else float(sim["start_condition"])
    rows = [{
        "stop_order": 0, "airport": sim["path"][0],
        "p10": start_delay, "p50": start_delay, "p90": start_delay,
        "prob_exceed_15": float(start_delay > 15), "is_fallback": False,
    }]
    for i, r in enumerate(sim["results"], start=1):
        s = summarize(r)
        rows.append({
            "stop_order": i, "airport": s["airport"],
            "p10": s["p10"], "p50": s["p50"], "p90": s["p90"],
            "prob_exceed_15": s["prob_exceed_15"], "is_fallback": r.used_route_fallback,
        })
    chart_df = pd.DataFrame(rows)
    chart_df["stop_label"] = chart_df["stop_order"].astype(str) + ". " + chart_df["airport"]
    stop_order_labels = chart_df["stop_label"].tolist()

    fallback_hops = chart_df.loc[chart_df["is_fallback"], "airport"].tolist()
    if fallback_hops:
        st.info(
            f"Network-wide estimate used (no/thin direct route history) for the "
            f"hop(s) arriving at: {', '.join(fallback_hops)}."
        )

    band = alt.Chart(chart_df).mark_area(opacity=0.25).encode(
        x=alt.X("stop_label:N", sort=stop_order_labels, title="Stop"),
        y=alt.Y("p10:Q", title="Delay (minutes)"),
        y2=alt.Y2("p90:Q"),
    )
    median = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X("stop_label:N", sort=stop_order_labels, title="Stop"),
        y=alt.Y("p50:Q", title="Delay (minutes)"),
        tooltip=["airport", "p10", "p50", "p90",
                 alt.Tooltip("prob_exceed_15:Q", format=".0%", title="P(delay > 15min)"),
                 "is_fallback"],
    )
    st.altair_chart((band + median).properties(height=350), use_container_width=True)

    st.dataframe(
        chart_df[["stop_order", "airport", "p10", "p50", "p90", "prob_exceed_15", "is_fallback"]],
        use_container_width=True, hide_index=True,
    )


def render_tables(airports_f: pd.DataFrame, routes_f: pd.DataFrame):
    """Bottom section: a single Airports/Routes toggle (tabs) instead of
    two always-open tables, each supporting multi-row selection. Checked
    rows stage into pending_focus_airports/routes — same staged-not-live
    rule as the sidebar sliders, so checking a box alone can't re-filter
    the map or tables. Go (rendered above this, in main()) is what copies
    the pending selection into focus_airports/routes; the two lists are
    independent and combine (union) rather than one replacing the other.

    SEARCH + SELECTION: st.dataframe's row selection is POSITIONAL (it
    tracks "row 0, row 3, ..." against whatever data you last passed it),
    and that position carries over across reruns as long as the widget's
    key doesn't change — even after the search box changes what data is
    passed in. Left alone, that means clearing/changing the search box
    can silently re-map an old selected position onto a completely
    different row and re-stage the wrong airport/route. To avoid that:
      - the widget's key includes the search text, so every distinct
        search view is its own widget instance with its own positions —
        a stale position from a different view can never leak in
      - on a view's first render we pre-seed its selection (via
        st.session_state[key]) from pending_focus_*, so previously
        staged rows still show checked when you search back to them
      - pending_focus_* is updated by MERGING: only rows currently
        visible under the search have their staged status replaced by
        what's checked in view; anything staged but hidden by the
        current search is left untouched rather than being dropped
    The "Staged" column mirrors true pending_focus_* membership on every
    render (not the widget's own highlight), so it stays correct even in
    the split second before a stale-position case above would otherwise
    show the wrong thing. Column-header click-to-sort is native to
    st.dataframe — no extra code needed for that part."""
    st.subheader("Data")
    reset = st.session_state["table_reset"]
    tab_airports, tab_routes = st.tabs(["Airports", "Routes"])

    with tab_airports:
        search = st.text_input("Search airport code", key=f"airport_search_{reset}").strip().upper()
        shown = airports_f
        if search:
            shown = shown[shown["airport"].str.contains(search)]
        shown = shown.reset_index(drop=True)

        pending_airports = st.session_state["pending_focus_airports"]
        table_key = f"airport_table_{reset}_{search}"
        if table_key not in st.session_state:
            preselected = [i for i, code in enumerate(shown["airport"]) if code in pending_airports]
            st.session_state[table_key] = {"selection": {"rows": preselected, "columns": []}}

        shown_display = shown.copy()
        shown_display.insert(0, "Staged", shown_display["airport"].isin(pending_airports))
        event = st.dataframe(
            shown_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="multi-row",
            key=table_key,
            column_config={
                "Staged": st.column_config.CheckboxColumn(disabled=True, help="Staged — click Go to apply"),
            },
        )
        checked = resolve_selection(shown, event.selection.rows)
        checked_codes = set(checked["airport"]) if checked is not None else set()
        visible_codes = set(shown["airport"])
        new_pending = (set(pending_airports) - visible_codes) | checked_codes
        if new_pending != set(pending_airports):
            st.session_state["pending_focus_airports"] = list(new_pending)
            st.rerun()

    with tab_routes:
        search = st.text_input("Search origin or destination", key=f"route_search_{reset}").strip().upper()
        shown = routes_f
        if search:
            shown = shown[shown["origin"].str.contains(search) | shown["dest"].str.contains(search)]
        shown = shown.reset_index(drop=True)

        pending_routes = st.session_state["pending_focus_routes"]
        pending_routes_set = set(pending_routes)
        table_key = f"route_table_{reset}_{search}"
        if table_key not in st.session_state:
            preselected = [
                i for i, pair in enumerate(zip(shown["origin"], shown["dest"])) if pair in pending_routes_set
            ]
            st.session_state[table_key] = {"selection": {"rows": preselected, "columns": []}}

        shown_display = shown.copy()
        pairs = pd.Series(list(zip(shown_display["origin"], shown_display["dest"])), index=shown_display.index)
        shown_display.insert(0, "Staged", pairs.isin(pending_routes_set))
        event = st.dataframe(
            shown_display, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="multi-row",
            key=table_key,
            column_config={
                "Staged": st.column_config.CheckboxColumn(disabled=True, help="Staged — click Go to apply"),
            },
        )
        checked = resolve_selection(shown, event.selection.rows)
        checked_pairs = set(zip(checked["origin"], checked["dest"])) if checked is not None else set()
        visible_pairs = set(zip(shown["origin"], shown["dest"]))
        new_pending = (pending_routes_set - visible_pairs) | checked_pairs
        if new_pending != pending_routes_set:
            st.session_state["pending_focus_routes"] = list(new_pending)
            st.rerun()


if __name__ == "__main__":
    main()