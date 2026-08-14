"""
app.py

Streamlit + pydeck dashboard showing which airports amplify or absorb delay
(node effect) and which routes gain or lose delay in the air (edge effect).

Reads ONLY map_data.duckdb — the small pre-aggregated export produced by
build_map_export.py. Never touches the full flights.duckdb / raw pipeline,
by design: this file is what actually deploys.

Run locally:   streamlit run app.py
Deploy:        push this repo (with map_data.duckdb committed) to GitHub,
               then point Streamlit Community Cloud at it. See README for
               the full deployment checklist.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pydeck as pdk
import streamlit as st

DB_PATH = Path(__file__).parent / "map_data.duckdb"

st.set_page_config(page_title="Delay Propagation Map", layout="wide")


# ----------------------------------------------------------------------
# DATA LOADING — cached so repeated interactions don't re-hit the file.
# read_only=True: this app never writes, and read-only avoids any lock
# contention if you ever run multiple instances against the same file.
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


# ----------------------------------------------------------------------
# COLOR MAPPING — diverging scale: red = amplifies delay, blue = absorbs
# delay, computed once per load rather than re-derived on every rerun.
# ----------------------------------------------------------------------

RED = (220, 60, 40)
BLUE = (40, 110, 220)
GRAY = (160, 160, 160)


def diverging_color(value: float, scale: float, alpha: int = 200) -> list:
    """Map a signed value to an RGBA color: red (positive) <-> blue
    (negative), intensity scaled by magnitude relative to `scale`
    (the value that maps to full saturation)."""
    if pd.isna(value):
        return list(GRAY) + [alpha]
    t = max(-1.0, min(1.0, value / scale)) if scale else 0.0
    if t >= 0:
        r = GRAY[0] + (RED[0] - GRAY[0]) * t
        g = GRAY[1] + (RED[1] - GRAY[1]) * t
        b = GRAY[2] + (RED[2] - GRAY[2]) * t
    else:
        t = -t
        r = GRAY[0] + (BLUE[0] - GRAY[0]) * t
        g = GRAY[1] + (BLUE[1] - GRAY[1]) * t
        b = GRAY[2] + (BLUE[2] - GRAY[2]) * t
    return [int(r), int(g), int(b), alpha]


# ----------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------

def main():
    if not DB_PATH.exists():
        st.error(
            f"map_data.duckdb not found next to app.py. Run "
            f"`python build_map_export.py` first, then commit the resulting "
            f"file alongside app.py."
        )
        st.stop()

    airports, routes = load_data()

    st.title("Delay Propagation Map")
    st.caption(
        "Airports colored by node effect (did the turn amplify or absorb "
        "delay); routes colored by edge effect (did the flight gain or "
        "lose delay in the air)."
    )

    with st.sidebar:
        st.header("Filters")
        min_turns = st.slider(
            "Minimum turn events (airports)", 0,
            int(airports["turn_events"].max()), 0,
            help="Hide airports with too few sampled turns to trust their average.",
        )
        min_flights = st.slider(
            "Minimum flights (routes)", 0,
            int(routes["flights"].max()), 0,
            help="Hide routes with too few flights to trust their average.",
        )
        show_routes = st.checkbox("Show routes", value=True)
        show_airports = st.checkbox("Show airports", value=True)

        st.divider()
        st.caption(
            "**Red** = amplifies delay &nbsp;&nbsp; **Blue** = absorbs delay "
            "&nbsp;&nbsp; **Gray** = near neutral"
        )

    airports_f = airports[airports["turn_events"] >= min_turns].copy()
    routes_f = routes[routes["flights"] >= min_flights].copy()

    if airports_f.empty and routes_f.empty:
        st.warning("No airports or routes match the current filters.")
        st.stop()

    # Color/size scales derived from the filtered data so the palette
    # stays meaningful as filters change.
    node_scale = max(airports_f["avg_node_effect"].abs().max(), 1) if not airports_f.empty else 1
    edge_scale = max(routes_f["avg_edge_effect_delay"].abs().max(), 1) if not routes_f.empty else 1

    airports_f["color"] = airports_f["avg_node_effect"].apply(
        lambda v: diverging_color(v, node_scale)
    )
    airports_f["radius"] = 8000 + (airports_f["turn_events"] / airports_f["turn_events"].max()) * 35000

    routes_f["color"] = routes_f["avg_edge_effect_delay"].apply(
        lambda v: diverging_color(v, edge_scale, alpha=160)
    )
    routes_f["width"] = 1 + (routes_f["flights"] / routes_f["flights"].max()) * 6

    layers = []
    if show_routes and not routes_f.empty:
        layers.append(pdk.Layer(
            "ArcLayer",
            data=routes_f,
            get_source_position=["origin_lon", "origin_lat"],
            get_target_position=["dest_lon", "dest_lat"],
            get_source_color="color",
            get_target_color="color",
            get_width="width",
            pickable=True,
            auto_highlight=True,
        ))
    if show_airports and not airports_f.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=airports_f,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
        ))

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

    st.pydeck_chart(deck, use_container_width=True, height=650)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Airports shown", len(airports_f))
    col2.metric("Routes shown", len(routes_f))
    amplifying = int((airports_f["avg_node_effect"] > 0).sum())
    absorbing = int((airports_f["avg_node_effect"] < 0).sum())
    col3.metric("Amplifying airports", amplifying)
    col4.metric("Absorbing airports", absorbing)

    with st.expander("Airport data"):
        st.dataframe(airports_f.drop(columns=["color", "radius"]), use_container_width=True)
    with st.expander("Route data"):
        st.dataframe(routes_f.drop(columns=["color", "width"]), use_container_width=True)


if __name__ == "__main__":
    main()