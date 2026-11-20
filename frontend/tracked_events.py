
import streamlit as st
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = (
    PROJECT_ROOT / "outputs" / "anomaly_results"
)

EVENT_FILE = OUTPUT_FOLDER / "tracked_weather_events.csv"

ISOLATED_FILE = OUTPUT_FOLDER / "isolated_weather_anomalies.csv"


# --------------------------------------------------
# LOAD EVENT DATA
# --------------------------------------------------

@st.cache_data
def load_event_data():

    if not EVENT_FILE.exists():
        return None, None

    events = pd.read_csv(EVENT_FILE)

    isolated = None

    if ISOLATED_FILE.exists():
        isolated = pd.read_csv(ISOLATED_FILE)

    return events, isolated


# --------------------------------------------------
# MAIN DASHBOARD FUNCTION
# --------------------------------------------------

def render_tracked_events():

    st.header("🌪️ Tracked Weather Events")

    st.write(
        "This section displays candidate weather anomaly "
        "groups identified by the GNN event-tracking module."
    )

    st.info(
        "These are model-generated candidate events, "
        "not independently confirmed meteorological events."
    )

    events, isolated = load_event_data()

    if events is None:

        st.warning(
            "Tracked event data was not found. "
            "Please run backend/event_tracker.py first."
        )

        return

    if events.empty:

        st.warning("No tracked event records are available.")

        return

    # --------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------

    required_columns = [
        "event_id",
        "start_time",
        "end_time",
        "duration_hours",
        "anomaly_count",
        "num_locations",
        "location_names",
        "max_gnn_score",
        "severity",
        "dominant_signal",
        "event_category"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:

        st.error(
            "The event CSV is missing expected columns: "
            + ", ".join(missing_columns)
        )

        st.write("Available columns:", list(events.columns))

        return

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    events = events.copy()

    events["start_time"] = pd.to_datetime(
        events["start_time"],
        errors="coerce"
    )

    events["end_time"] = pd.to_datetime(
        events["end_time"],
        errors="coerce"
    )

    events["duration_hours"] = pd.to_numeric(
        events["duration_hours"],
        errors="coerce"
    )

    events["anomaly_count"] = pd.to_numeric(
        events["anomaly_count"],
        errors="coerce"
    )

    events["num_locations"] = pd.to_numeric(
        events["num_locations"],
        errors="coerce"
    )

    events["max_gnn_score"] = pd.to_numeric(
        events["max_gnn_score"],
        errors="coerce"
    )

    # --------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------

    st.subheader("📊 Event Summary")

    total_events = len(events)

    multi_record = (
        events["event_category"]
        .astype(str)
        .str.contains("Multi-record", case=False)
        .sum()
    )

    isolated_count = (
        events["event_category"]
        .astype(str)
        .str.contains("Isolated", case=False)
        .sum()
    )

    affected_locations = set()

    for value in events["location_names"].dropna():

        for location in str(value).split(","):
            location = location.strip()

            if location:
                affected_locations.add(location)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Candidate Events",
        total_events
    )

    col2.metric(
        "Multi-record Events",
        int(multi_record)
    )

    col3.metric(
        "Isolated Detections",
        int(isolated_count)
    )

    col4.metric(
        "Affected Locations",
        len(affected_locations)
    )

    st.divider()

    # --------------------------------------------------
    # FILTERS
    # --------------------------------------------------

    st.subheader("🔎 Filter Events")

    col1, col2 = st.columns(2)

    severity_options = sorted(
        events["severity"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    category_options = sorted(
        events["event_category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    with col1:

        selected_severity = st.multiselect(
            "Select severity",
            severity_options,
            default=severity_options
        )

    with col2:

        selected_category = st.multiselect(
            "Select event category",
            category_options,
            default=category_options
        )

    filtered = events[
        events["severity"].astype(str).isin(
            selected_severity
        )
        &
        events["event_category"].astype(str).isin(
            selected_category
        )
    ].copy()

    st.write(
        f"Displaying {len(filtered)} of {total_events} events."
    )

    st.divider()

    # --------------------------------------------------
    # SEVERITY CHART
    # --------------------------------------------------

    st.subheader("⚠️ Event Severity Distribution")

    if not filtered.empty:

        severity_counts = (
            filtered["severity"]
            .value_counts()
        )

        st.bar_chart(
            severity_counts,
            width="stretch"
        )

    else:

        st.info("No events match the selected filters.")

    # --------------------------------------------------
    # DOMINANT WEATHER SIGNAL
    # --------------------------------------------------

    st.subheader("🌧️ Dominant Weather Signals")

    if not filtered.empty:

        signal_counts = (
            filtered["dominant_signal"]
            .fillna("Unknown")
            .value_counts()
        )

        st.bar_chart(
            signal_counts,
            width="stretch"
        )

    # --------------------------------------------------
    # EVENT TABLE
    # --------------------------------------------------

    st.subheader("📋 Candidate Event Records")

    display_columns = [
        "event_id",
        "start_time",
        "end_time",
        "duration_hours",
        "anomaly_count",
        "num_locations",
        "location_names",
        "max_gnn_score",
        "severity",
        "dominant_signal",
        "event_category"
    ]

    display_df = filtered[display_columns].copy()

    display_df = display_df.sort_values(
        "start_time",
        ascending=False
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------
    # EVENT DETAILS
    # --------------------------------------------------

    st.subheader("🔍 Inspect an Individual Event")

    if not filtered.empty:

        event_ids = filtered["event_id"].astype(str).tolist()

        selected_event_id = st.selectbox(
            "Choose an event ID",
            event_ids
        )

        selected_event = filtered[
            filtered["event_id"].astype(str)
            == selected_event_id
        ].iloc[0]

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:

            st.write("**Event ID:**", selected_event["event_id"])

            st.write(
                "**Start time:**",
                selected_event["start_time"]
            )

            st.write(
                "**End time:**",
                selected_event["end_time"]
            )

            st.write(
                "**Duration (hours):**",
                selected_event["duration_hours"]
            )

        with detail_col2:

            st.write(
                "**Severity:**",
                selected_event["severity"]
            )

            st.write(
                "**Anomaly count:**",
                selected_event["anomaly_count"]
            )

            st.write(
                "**Affected locations:**",
                selected_event["location_names"]
            )

            st.write(
                "**Dominant signal:**",
                selected_event["dominant_signal"]
            )

        st.write(
            "**Event category:**",
            selected_event["event_category"]
        )

        st.write(
            "**Maximum GNN score:**",
            selected_event["max_gnn_score"]
        )

    # --------------------------------------------------
    # ISOLATED DETECTIONS
    # --------------------------------------------------

    st.divider()

    st.subheader("📍 Isolated Weather Anomaly Detections")

    if isolated is not None and not isolated.empty:

        st.write(
            "These detections were not grouped into "
            "multi-record candidate events."
        )

        st.dataframe(
            isolated,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No separate isolated detection records were found."
        )

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------

    st.divider()

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Event Records",
        data=csv_data,
        file_name="filtered_weather_events.csv",
        mime="text/csv"
    )

    st.success(
        "Tracked weather event data loaded successfully."
    )
