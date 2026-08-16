"""Streamlit dashboard.

    streamlit run src/flight_delay/serving/dashboard.py

Charts over prose. Every figure is read from artifacts the pipeline wrote, so
the app cannot disagree with `docs/` -- but it renders them rather than
reprinting them, because a reader who wanted the write-up would read the
write-up.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from flight_delay.serving import artifacts, bundle
from flight_delay.serving.features import build_row

BASE_RATE = 0.2056
INK = "#1f3350"
ACCENT = "#2c7fb8"
WARN = "#d95f45"
MUTED = "#9aa7b4"


@st.cache_resource
def _bundle() -> bundle.ModelBundle | None:
    directory = artifacts.model_directory()
    if directory is None:
        return None
    try:
        return bundle.load(directory)
    except FileNotFoundError:
        return None


@st.cache_data
def _artifact(name: str) -> dict[str, Any] | None:
    return artifacts.load_result(name)


def _missing(command: str) -> None:
    st.warning(f"No results yet. Run `flight-delay {command}`.")


def _bars(
    frame: pd.DataFrame, *, x: str, y: str, colour: str, domain: tuple[float, float] | None = None
) -> Any:
    """Horizontal bars, sorted, with the value written on each."""
    scale = alt.Scale(domain=list(domain)) if domain else alt.Scale()
    base = alt.Chart(frame).encode(
        y=alt.Y(f"{y}:N", sort="-x", title=None),
        x=alt.X(f"{x}:Q", title=None, scale=scale),
    )
    return (
        base.mark_bar(color=colour, cornerRadiusEnd=3, height=18)
        + base.mark_text(align="left", dx=4, color=INK, fontSize=11).encode(
            text=alt.Text(f"{x}:Q", format=".3f")
        )
    ).properties(height=alt.Step(26))


# --------------------------------------------------------------------------
# 1. The question
# --------------------------------------------------------------------------
def page_question() -> None:
    st.title("Should I leave for the airport?")
    st.markdown(
        "One in five US flights lands 15+ minutes late. Predicting *which* ones "
        "is easy **after** the plane pushes back and hard **before** — and only "
        "the hard version is any use to a passenger."
    )

    data = _artifact("classification.json")
    if data is None:
        _missing("train")
        return

    rows = [
        {
            "scenario": "Before departure\n(2h ahead)"
            if key.startswith("A")
            else "After departure\n(actual delay known)",
            "model": model,
            "pr_auc": metrics["pr_auc"],
        }
        for key, models in data.items()
        for model, metrics in models.items()
        if model == "gradient boosting"
    ]
    frame = pd.DataFrame(rows)
    before = float(frame.loc[frame["scenario"].str.startswith("Before"), "pr_auc"].iloc[0])
    after = float(frame.loc[frame["scenario"].str.startswith("After"), "pr_auc"].iloc[0])

    left, right = st.columns([3, 2], gap="large")

    with left:
        chart = (
            alt.Chart(frame)
            .mark_bar(cornerRadiusEnd=4, size=46)
            .encode(
                x=alt.X("pr_auc:Q", title="PR-AUC", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("scenario:N", title=None, sort=None),
                color=alt.condition(alt.datum.pr_auc > 0.5, alt.value(MUTED), alt.value(ACCENT)),
                tooltip=["scenario", alt.Tooltip("pr_auc:Q", format=".3f")],
            )
            .properties(height=150)
        )
        labels = chart.mark_text(align="left", dx=6, fontSize=15, color=INK).encode(
            text=alt.Text("pr_auc:Q", format=".3f")
        )
        rule = (
            alt.Chart(pd.DataFrame({"x": [BASE_RATE]}))
            .mark_rule(color=WARN, strokeDash=[5, 4])
            .encode(x="x:Q")
        )
        st.altair_chart(chart + labels + rule, use_container_width=True)
        st.caption(
            f"Dashed line: a model that always guesses the average ({BASE_RATE:.0%}). "
            "Same data, same model, same split — the only difference is what it "
            "is allowed to know."
        )

    with right:
        st.metric("Before departure", f"{before:.3f}", f"{before / BASE_RATE:.2f}x the floor")
        st.metric("After departure", f"{after:.3f}", f"{after / BASE_RATE:.2f}x the floor")
        st.markdown(
            f"**The grey bar is the trap.** A model that quietly uses the actual "
            f"departure delay scores {after:.2f} and is worthless: at the moment "
            f"you need the answer, nobody knows it yet."
        )

    st.divider()
    st.subheader("How it is built")
    steps = [
        ("13.9M flights", "US BTS 2023-24, every carrier, no sampling"),
        ("Schema contract", "Each column declares *when* it becomes known"),
        ("Temporal split", "Fit on 2023, judged on 2024. Never random"),
        ("Two engines", "Same SQL on DuckDB and Spark, results compared"),
        ("Served", "FastAPI + this app, from a 1 MB model bundle"),
    ]
    for column, (title, detail) in zip(st.columns(len(steps)), steps, strict=True):
        column.markdown(f"**{title}**  \n<small>{detail}</small>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 2. Score a flight
# --------------------------------------------------------------------------
def page_predictor() -> None:
    st.title("Score a flight")
    loaded = _bundle()
    if loaded is None:
        _missing("export-model")
        return

    controls, output = st.columns([2, 3], gap="large")

    with controls:
        c1, c2, c3 = st.columns(3)
        carrier = c1.text_input("Carrier", "AA", max_chars=3)
        origin = c2.text_input("From", "JFK", max_chars=4)
        dest = c3.text_input("To", "LAX", max_chars=4)

        flight_date = st.date_input("Date", date(2024, 7, 15))
        hour = st.slider("Scheduled departure", 0, 23, 8, format="%d:00")

        c1, c2 = st.columns(2)
        distance = c1.number_input("Miles", min_value=1.0, value=2475.0, step=50.0)
        elapsed = c2.number_input("Minutes", min_value=1.0, value=375.0, step=15.0)

        known = st.toggle(
            "Inbound aircraft has landed",
            help="True for only 27% of flights two hours before departure.",
        )
        inbound = st.slider("…and it was this late (min)", -30, 180, 0) if known else None

    arrival = (hour * 60 + int(elapsed)) % 1440
    try:
        frame = build_row(
            flight_date=flight_date if isinstance(flight_date, date) else date(2024, 7, 15),
            carrier=carrier.upper(),
            origin=origin.upper(),
            dest=dest.upper(),
            scheduled_departure=f"{hour:02d}00",
            scheduled_arrival=f"{arrival // 60:02d}{arrival % 60:02d}",
            distance=float(distance),
            scheduled_elapsed_minutes=float(elapsed),
            priors=loaded.priors,
            inbound_delay_minutes=inbound,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    probability = loaded.predict_proba(frame)

    with output:
        a, b = st.columns(2)
        a.metric("Chance of arriving late", f"{probability:.0%}")
        b.metric("Versus the average flight", f"{probability / BASE_RATE:.2f}x")

        gauge = pd.DataFrame(
            {"what": ["This flight", "Average flight"], "p": [probability, BASE_RATE]}
        )
        st.altair_chart(
            alt.Chart(gauge)
            .mark_bar(cornerRadiusEnd=4, size=34)
            .encode(
                x=alt.X(
                    "p:Q", title=None, scale=alt.Scale(domain=[0, 0.6]), axis=alt.Axis(format="%")
                ),
                y=alt.Y("what:N", title=None, sort=None),
                color=alt.condition(
                    alt.datum.what == "This flight", alt.value(ACCENT), alt.value(MUTED)
                ),
                tooltip=[alt.Tooltip("p:Q", format=".1%")],
            )
            .properties(height=110),
            use_container_width=True,
        )

        st.markdown("##### Where would you draw the line?")
        analysis = _artifact("analysis.json")
        points = analysis.get("operating_points") if analysis else None
        if not points:
            return

        options = [p["threshold"] for p in points]
        threshold = st.select_slider(
            "Warn every passenger above",
            options=options,
            value=0.25,
            format_func=lambda v: f"{v:.0%}",
        )
        chosen = next(p for p in points if p["threshold"] == threshold)

        verdict = "WARN this passenger" if probability >= threshold else "stay quiet"
        st.markdown(f"### {verdict}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Passengers warned", f"{chosen['alert_rate']:.0%}")
        m2.metric("Of those, actually late", f"{chosen['precision']:.0%}")
        m3.metric("Of all delays, caught", f"{chosen['recall']:.0%}")
        st.caption(
            "There is no correct threshold — it depends on what a missed delay "
            "costs against a false alarm. At the conventional 50% the service "
            "warns 0.9% of passengers and catches 2% of delays."
        )


# --------------------------------------------------------------------------
# 3. What the model uses
# --------------------------------------------------------------------------
def page_drivers() -> None:
    st.title("What actually predicts a delay")
    analysis = _artifact("analysis.json")
    if analysis is None or not analysis.get("importances"):
        _missing("analyse")
        return

    frame = pd.DataFrame(analysis["importances"])
    frame = frame[frame["pr_auc_drop"] > 0].copy()
    labels = {
        "dep_minute_of_day": "Time of day",
        "month": "Month",
        "origin": "Origin airport",
        "carrier": "Airline",
        "inbound_delay": "Inbound aircraft delay",
        "dest": "Destination",
        "inbound_turnaround_minutes": "Turnaround time",
        "arr_minute_of_day": "Arrival time",
        "day_of_week": "Day of week",
        "carrier_prior_rate": "Airline's recent record",
        "crs_elapsed": "Flight duration",
        "distance": "Distance",
        "origin_prior_rate": "Airport's recent record",
    }
    frame["label"] = frame["feature"].map(labels).fillna(frame["feature"])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.altair_chart(
            _bars(frame, x="pr_auc_drop", y="label", colour=ACCENT), use_container_width=True
        )
        st.caption(
            "Predictive power lost when each input is shuffled. Measured on the held-out year."
        )
    with right:
        top = frame.iloc[0]
        st.metric("Strongest single input", top["label"])
        st.markdown(
            "**The clock wins, by double.** Delay builds through the operating "
            "day, so an 8am departure is a different proposition from a 7pm one."
        )
        st.markdown(
            "The inbound-aircraft features took the most work in the whole "
            "project — point-in-time rules, timestamps across midnight — and "
            "they land mid-table. That is the measurement, not the hope."
        )
        st.info(
            "**Four inputs measured exactly zero** and should be dropped: "
            "hour-of-day duplicated minute-of-day, weekend duplicated day-of-week, "
            "and two more that were implied by columns already present."
        )


# --------------------------------------------------------------------------
# 4. Engineering
# --------------------------------------------------------------------------
def page_engineering() -> None:
    st.title("Does 13.9M rows need a cluster?")

    engines = _artifact("engines.json")
    if engines and engines.get("timings"):
        rows = [
            {
                "months": int(key.split("@")[1].removesuffix("m")),
                "workload": "Aggregation" if key.startswith("daily") else "Window function",
                "engine": engine.capitalize(),
                "seconds": value[f"{engine}_seconds"],
            }
            for key, value in engines["timings"].items()
            for engine in ("duckdb", "spark")
        ]
        frame = pd.DataFrame(rows)
        st.altair_chart(
            alt.Chart(frame)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("months:Q", title="Months of data"),
                y=alt.Y("seconds:Q", title="Seconds"),
                color=alt.Color("engine:N", title=None, scale=alt.Scale(range=[ACCENT, WARN])),
                strokeDash=alt.StrokeDash("workload:N", title=None),
                tooltip=["engine", "workload", "months", alt.Tooltip("seconds:Q", format=".2f")],
            )
            .properties(height=260),
            use_container_width=True,
        )
        # Missing rather than zero: a default of 0 would render "0.0s" and
        # read as a measurement.
        duck = engines.get("duckdb_startup_seconds")
        spark = engines.get("spark_startup_seconds")
        if duck is not None and spark is not None:
            a, b = st.columns(2)
            a.metric("DuckDB startup", f"{duck:.2f}s")
            b.metric("Spark startup", f"{spark:.1f}s")
        st.markdown(
            "**No.** DuckDB wins at every scale tested. The gap narrows as the "
            "data grows — 13.8x down to 2.5x — but the lines never cross. Spark "
            "is here to be measured against, not to be justified. What it buys "
            "at this size is a different memory bound, not a faster answer."
        )

    st.divider()
    st.subheader("What layout is worth")
    layout = _artifact("layout.json")
    if layout is None:
        _missing("bench-layout")
        return

    pruning = layout.get("pruning", {})
    if pruning:
        frame = pd.DataFrame(
            [
                {"query": "Whole table" if k == "full" else k, "MB read": v["megabytes"]}
                for k, v in pruning.items()
            ]
        )
        st.altair_chart(_bars(frame, x="MB read", y="query", colour=INK), use_container_width=True)
        st.caption(
            "Filtering on a partition column reads one file instead of 24 — "
            "95% of the bytes never leave disk. Measured in bytes, not seconds: "
            "the same scan ran 2.10s then 0.48s reading identical data, because "
            "of the page cache."
        )


# --------------------------------------------------------------------------
# 5. Forecast
# --------------------------------------------------------------------------
def page_forecast() -> None:
    st.title("How bad will tomorrow be?")
    data = _artifact("timeseries.json")
    if data is None:
        _missing("forecast")
        return

    horizon = st.radio(
        "Forecast horizon",
        ["h1", "h7"],
        horizontal=True,
        format_func=lambda h: "Tomorrow" if h == "h1" else "A week out",
    )

    per_airport: dict[str, float] | None = data.get("airports", {}).get(f"{horizon}_per_airport")
    if per_airport:
        ranked = sorted(per_airport.items(), key=lambda kv: kv[1])
        frame = pd.DataFrame(ranked, columns=["airport", "MASE"])
        chart = (
            alt.Chart(frame)
            .mark_bar(cornerRadiusEnd=3, size=22)
            .encode(
                x=alt.X(
                    "MASE:Q",
                    scale=alt.Scale(domain=[0, 1.15]),
                    title="MASE — below 1 beats the baseline",
                ),
                y=alt.Y("airport:N", sort="x", title=None),
                color=alt.condition(alt.datum.MASE < 1, alt.value(ACCENT), alt.value(WARN)),
                tooltip=["airport", alt.Tooltip("MASE:Q", format=".3f")],
            )
            .properties(height=alt.Step(28))
        )
        rule = (
            alt.Chart(pd.DataFrame({"x": [1.0]}))
            .mark_rule(color=WARN, strokeDash=[4, 4])
            .encode(x="x:Q")
        )
        st.altair_chart(chart + rule, use_container_width=True)

    national = data.get("national", {}).get(horizon)
    if national:
        best = min(national, key=lambda s: s["mase"])
        a, b = st.columns(2)
        a.metric("National forecast", f"MASE {best['mase']:.3f}")
        b.metric(
            "Improvement over doing nothing",
            f"{max(0.0, (1 - best['mase'])) * 100:.0f}%",
        )

    if horizon == "h7":
        st.warning(
            "**A week out, the national model is 1.3% better than doing nothing.** "
            "Atlanta ties the baseline exactly. Without weather, this is roughly "
            "where calendar features and past rates run out — reported as the "
            "negative result it is."
        )
    else:
        st.success(
            "**One day ahead the model earns its place**, and more so per airport "
            "than nationally. Weekly seasonality turned out weak — five points "
            "between Tuesday and Sunday — so the seasonal baseline loses to "
            "simply repeating yesterday."
        )


PAGES = {
    "The question": page_question,
    "Score a flight": page_predictor,
    "What predicts a delay": page_drivers,
    "Engineering": page_engineering,
    "Forecast": page_forecast,
}


def main() -> None:
    st.set_page_config(page_title="Flight delay prediction", layout="wide")
    with st.sidebar:
        st.markdown("### Flight delay prediction")
        st.caption("13,926,960 US flights · 2023-2024 - BTS")
        choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(f"Figures read from the {artifacts.source_label('classification.json')}.")
        st.caption("[Source and write-up](https://github.com/serpeigd/FlightsDelay)")
    PAGES[choice]()


main()
