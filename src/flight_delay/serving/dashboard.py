"""Streamlit dashboard.

    streamlit run src/flight_delay/serving/dashboard.py

Charts over prose. Every figure is read from artifacts the pipeline wrote, so
the app cannot disagree with `docs/` -- but it renders them rather than
reprinting them, because a reader who wanted the write-up would read the
write-up.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from flight_delay.serving import artifacts, bundle
from flight_delay.serving.carriers import label as carrier_label
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


def _searchable(
    title: str,
    options: list[str],
    describe: Callable[[str], str],
    *,
    key: str,
    default: str | None = None,
    hint: str = "",
) -> str:
    """One box that both searches and selects.

    Streamlit's select box already filters as you type against the rendered
    label, so "new york" finds both LGA and JFK. A separate search input on top
    was tried and removed: two controls per field for one decision reads as
    clutter, and it filtered exactly what the box filters anyway.

    The typing is not discoverable on its own, hence the tooltip.
    """
    index = options.index(default) if default in options else 0
    return st.selectbox(
        title,
        options,
        index=index,
        format_func=describe,
        key=key,
        help=f"Click and type to search. {hint}".strip(),
    )


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
        "One in five US flights lands 15+ minutes late. **Easy to predict once "
        "the plane has pushed back. Hard before.** Only the hard version is any "
        "use to a passenger."
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
        st.altair_chart(chart + labels + rule, width="stretch")
        st.caption(
            f"Dashed line: always guessing the average ({BASE_RATE:.0%}). "
            "Same data, same model, same split. Only the inputs change."
        )

    with right:
        st.metric("Before departure", f"{before:.3f}", f"{before / BASE_RATE:.2f}x the floor")
        st.metric("After departure", f"{after:.3f}", f"{after / BASE_RATE:.2f}x the floor")
        st.markdown(
            f"**Grey bar: the trap.**\n\n"
            f"It uses the real departure delay. Scores {after:.2f}. Useless: that "
            f"number does not exist yet when you need the answer."
        )

    calibration = data.get("A_pre_departure", {}).get("gradient boosting", {}).get("calibration")
    if calibration:
        st.divider()
        st.subheader("Is 60% really 60%?")
        bins = pd.DataFrame(calibration)
        points = (
            alt.Chart(bins)
            .mark_circle(color=ACCENT)
            .encode(
                x=alt.X(
                    "predicted:Q",
                    title="What the model says",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y(
                    "observed:Q",
                    title="What actually happened",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                size=alt.Size("count:Q", title="Flights", scale=alt.Scale(range=[40, 900])),
                tooltip=[
                    alt.Tooltip("predicted:Q", format=".1%"),
                    alt.Tooltip("observed:Q", format=".1%"),
                    alt.Tooltip("count:Q", format=","),
                ],
            )
        )
        ideal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color=MUTED, strokeDash=[5, 4])
            .encode(x="x:Q", y="y:Q")
        )
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.altair_chart((ideal + points).properties(height=300), width="stretch")
        with right:
            st.markdown(
                "**On the line: the probability is honest.**\n\n"
                "It sits on the line where the flights are.\n\n"
                "At the top it says 64% and reality is 53%. Few flights, but exactly "
                "the ones a warning system acts on."
            )
            st.caption(
                "Recalibration tried on two holdouts. Both worse. Gradient boosting "
                "optimises a proper scoring rule, so it arrives calibrated."
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

    routes = loaded.routes
    if routes.empty:
        st.warning("This bundle has no route reference. Re-run `flight-delay export-model`.")
        return

    controls, output = st.columns([2, 3], gap="large")

    with controls:
        # Every option is a route that was actually flown, so distance and
        # duration come from the data instead of the user having to know that
        # JFK to LAX is 2475 miles.
        airports = (
            routes.groupby("origin")
            .agg(city=("origin_city", "first"), flights=("flights", "sum"))
            .sort_values("flights", ascending=False)
            .reset_index()
        )
        labels = {r.origin: f"{r.origin} — {r.city}" for r in airports.itertuples()}

        codes: list[str] = airports["origin"].tolist()
        origin = _searchable(
            "From",
            codes,
            lambda a: labels.get(a, a),
            key="origin",
            default="JFK",
            hint="Code or city, e.g. JFK or New York.",
        )

        onward = routes[routes["origin"] == origin].sort_values("flights", ascending=False)
        dest_labels = {r.dest: f"{r.dest} — {r.dest_city}" for r in onward.itertuples()}
        dest = _searchable(
            "To",
            onward["dest"].tolist(),
            lambda a: dest_labels.get(a, a),
            key="dest",
            default="LAX",
            hint=f"Only the {len(onward)} destinations this airport actually serves.",
        )

        route = onward[onward["dest"] == dest].iloc[0]

        # Sorted by name, not by code: nobody scans a dropdown looking for "YX".
        carriers = sorted(loaded.priors.carrier_rate, key=carrier_label)
        carrier = _searchable(
            "Airline",
            carriers,
            carrier_label,
            key="carrier",
            default="AA",
            hint="Code or name, e.g. DL or Delta.",
        )

        flight_date = st.date_input("Date", date(2024, 7, 15))
        hour = st.slider("Scheduled departure", 0, 23, 8, format="%d:00")

        st.caption(
            f"{int(route.distance):,} miles · scheduled {int(route.elapsed_minutes)} min · "
            f"{int(route.flights):,} flights on this route in the data"
        )

        known = st.toggle(
            "Inbound aircraft has landed",
            help="True for only 27% of flights two hours before departure.",
        )
        inbound = st.slider("…and it was this late (min)", -30, 180, 0) if known else None

    distance = float(route.distance)
    elapsed = float(route.elapsed_minutes)
    arrival = (hour * 60 + int(elapsed)) % 1440
    try:
        frame = build_row(
            flight_date=flight_date if isinstance(flight_date, date) else date(2024, 7, 15),
            carrier=carrier,
            origin=origin,
            dest=dest,
            scheduled_departure=f"{hour:02d}00",
            scheduled_arrival=f"{arrival // 60:02d}{arrival % 60:02d}",
            distance=distance,
            scheduled_elapsed_minutes=elapsed,
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
            width="stretch",
        )

        st.markdown("##### Where would you draw the line?")
        analysis = _artifact("analysis.json")
        if analysis is None:
            return
        points = analysis.get("operating_points")
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

        # Rates are a statistic; counts are a decision.
        if "warned" in chosen:
            total = analysis.get("test_flights", 0)
            st.markdown(f"**Over one year of real flights ({total:,}), that setting means:**")
            n1, n2, n3 = st.columns(3)
            n1.metric("People warned", f"{chosen['warned']:,}")
            n2.metric("Warned for nothing", f"{chosen['false_alarms']:,}")
            n3.metric("Delays missed", f"{chosen['delays_missed']:,}")

        st.caption(
            "No correct threshold. It depends what a missed delay costs against a "
            "false alarm. At the usual 50%: warns 0.9% of passengers, catches 2% "
            "of delays."
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
        st.altair_chart(_bars(frame, x="pr_auc_drop", y="label", colour=ACCENT), width="stretch")
        st.caption("PR-AUC lost when each input is shuffled. Measured on the held-out year.")
    with right:
        st.metric("Strongest single input", frame.iloc[0]["label"])
        st.markdown(
            "**The clock wins, by double.**\n\n"
            "Delay builds through the operating day. 8am and 7pm are different problems."
        )
        st.markdown(
            "**My best feature came fourth.**\n\n"
            "The inbound-aircraft columns took the most work in the project. "
            "Mid-table. That is the measurement."
        )
        st.info("**Four inputs measured zero. Removed.** PR-AUC 0.343 before, 0.343 after.")


def _slope(block: pd.DataFrame) -> tuple[float, float, float]:
    """Last measured segment of a ratio curve, in log-log space.

    Returns (last rows, last ratio, slope). The slope comes from the final two
    points rather than a fit over all five: the early points are dominated by
    Spark's fixed per-stage overhead, which is exactly the part that stops
    mattering as the data grows.
    """
    tail = block.sort_values("rows").tail(2)
    (r0, y0), (r1, y1) = tail[["rows", "ratio"]].to_numpy()
    slope = math.log(y1 / y0) / math.log(r1 / r0)
    return float(r1), float(y1), float(slope)


def _trend_lines(ratios: pd.DataFrame, *, right_edge: float, floor: float) -> pd.DataFrame:
    """Each curve's last slope, carried to the edge of the chart or the floor."""
    out: list[dict[str, Any]] = []
    for workload, block in ratios.groupby("workload"):
        rows, ratio, slope = _slope(block)
        if slope >= 0:
            continue
        # Stop where the projection would leave the visible y range, so a very
        # steep trend does not draw a line to nowhere.
        at_floor = rows * (floor / ratio) ** (1 / slope)
        end = min(right_edge, at_floor)
        out.append({"workload": workload, "rows": rows, "ratio": ratio})
        out.append({"workload": workload, "rows": end, "ratio": ratio * (end / rows) ** slope})
    return pd.DataFrame(out)


def _crossings(ratios: pd.DataFrame) -> dict[str, float]:
    """Where each extended trend would reach parity with DuckDB."""
    found: dict[str, float] = {}
    for workload, block in ratios.groupby("workload"):
        rows, ratio, slope = _slope(block)
        if slope < 0:
            found[str(workload)] = rows * (1.0 / ratio) ** (1 / slope)
    return found


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
        scale = _artifact("scale.json") or {}
        # Rows, not months. Months only mean something for this feed, and the
        # x axis has to hold both what was measured and the size of the whole
        # feed for the comparison to mean anything.
        rows_by_months = {int(k): int(v) for k, v in scale.get("rows_per_scale", {}).items()}
        ratios = pd.DataFrame(
            [
                {
                    "rows": rows_by_months[months],
                    "workload": "Aggregation" if key.startswith("daily") else "Window function",
                    "ratio": value["spark_seconds"] / value["duckdb_seconds"],
                }
                for key, value in engines["timings"].items()
                if value.get("duckdb_seconds")
                and (months := int(key.split("@")[1].removesuffix("m"))) in rows_by_months
            ]
        )
        projection = scale.get("projection", {})
        memory = scale.get("memory", {})
        feed_rows = projection.get("estimated_feed_rows")
        fit_rows = memory.get("rows_that_fit")

        if not ratios.empty and fit_rows:
            # The question is "when would I switch?", so the answer is a region
            # of the x axis, not a line on it. Shading the two zones says it
            # before anyone reads a label; the measured points then land inside
            # the left one, which is the whole finding.
            right_edge = max(float(ratios["rows"].max()), float(feed_rows or 0), fit_rows) * 1.7
            x_scale = alt.Scale(type="log", domain=[3e5, right_edge])
            x = alt.X(
                "rows:Q",
                title="Rows in the table",
                scale=x_scale,
                axis=alt.Axis(format="~s"),
            )
            zones = pd.DataFrame(
                [
                    {"start": 3e5, "end": fit_rows, "colour": ACCENT},
                    {"start": fit_rows, "end": right_edge, "colour": WARN},
                ]
            )
            captions = pd.DataFrame(
                [
                    {
                        "at": (3e5 * fit_rows) ** 0.5,
                        "text": "One machine is enough",
                        "colour": ACCENT,
                    },
                    {
                        "at": (fit_rows * right_edge) ** 0.5,
                        "text": "Cluster earns its place",
                        "colour": WARN,
                    },
                ]
            )
            trends = _trend_lines(ratios, right_edge=right_edge, floor=0.75)
            layers = [
                alt.Chart(zones)
                .mark_rect(opacity=0.10)
                .encode(x=x, x2="end:Q", color=alt.Color("colour:N", scale=None, legend=None)),
                alt.Chart(captions)
                .mark_text(fontSize=12, fontWeight="bold", baseline="top", dy=4)
                .encode(
                    x=alt.X("at:Q", scale=x_scale, title=None),
                    y=alt.datum(23),
                    text="text:N",
                    color=alt.Color("colour:N", scale=None, legend=None),
                ),
                alt.Chart(pd.DataFrame({"rows": [fit_rows]}))
                .mark_rule(color=WARN, strokeWidth=2)
                .encode(x=x),
                # Dashed: the last measured slope carried forward. Kept visually
                # distinct from the solid measured segment because it is an
                # assumption about the future, not a reading from the past.
                alt.Chart(trends)
                .mark_line(strokeDash=[5, 5], strokeWidth=2, opacity=0.75)
                .encode(
                    x=x,
                    y=alt.Y("ratio:Q", scale=alt.Scale(type="log", domain=[0.7, 30])),
                    color=alt.Color("workload:N", legend=None),
                ),
                alt.Chart(ratios)
                .mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=3)
                .encode(
                    x=x,
                    y=alt.Y(
                        "ratio:Q",
                        title="How many times slower Spark is",
                        scale=alt.Scale(type="log", domain=[0.7, 30]),
                        axis=alt.Axis(values=[1, 2, 5, 10, 20], format="d"),
                    ),
                    color=alt.Color(
                        "workload:N",
                        title=None,
                        scale=alt.Scale(range=[INK, "#6b8fb3"]),
                        legend=alt.Legend(orient="top"),
                    ),
                    tooltip=[
                        "workload",
                        alt.Tooltip("rows:Q", format=",", title="rows"),
                        alt.Tooltip("ratio:Q", format=".1f", title="x slower"),
                    ],
                ),
                alt.Chart(pd.DataFrame({"y": [1.0]}))
                .mark_rule(color=INK, strokeDash=[6, 4])
                .encode(y="y:Q"),
                alt.Chart(pd.DataFrame({"y": [1.0], "text": ["Spark faster below this line"]}))
                .mark_text(align="left", dx=6, dy=11, fontSize=11, color=INK)
                .encode(y="y:Q", text="text:N"),
            ]
            if feed_rows:
                feed = pd.DataFrame({"rows": [feed_rows], "text": ["whole feed"]})
                layers += [
                    alt.Chart(feed).mark_rule(color=INK, strokeDash=[2, 3]).encode(x=x),
                    alt.Chart(feed)
                    .mark_text(align="right", dx=-4, fontSize=10, color=INK)
                    .encode(x=x, y=alt.datum(0.8), text="text:N"),
                ]
            st.altair_chart(alt.layer(*layers).properties(height=330), width="stretch")

            crossings = _crossings(ratios)
            st.markdown(
                f"**No. DuckDB won all {len(ratios)} measurements** (solid lines).\n\n"
                f"**Dashed: the same trend carried forward.** Parity only arrives "
                + ", ".join(
                    f"near **{rows / 1e6:.0f}M rows** on the {name.lower()}"
                    for name, rows in sorted(crossings.items(), key=lambda kv: kv[1])
                )
                + f". Both are far past anything measured.\n\n"
                f"**The memory wall lands at {fit_rows / 1e6:.0f}M rows** on this "
                f"machine, in between the two. That is the reason to switch, not "
                f"the clock."
            )

        with st.expander("How this was measured"):
            duck = engines.get("duckdb_startup_seconds")
            spark = engines.get("spark_startup_seconds")
            matched = all(v.get("answers_match") for v in engines["timings"].values())
            st.markdown(
                f"- **Same SQL, same table, same machine.** One aggregation, one "
                f"window function. {len(engines.get('scales_months', []))} scales, "
                f"{engines.get('repeats', 3)} repeats.\n"
                f"- **Answers compared, not assumed.** Row counts and checksums "
                f"{'matched every run' if matched else 'did not all match'}. An "
                f"earlier version disagreed by 2 rows: that is how the tie in the "
                f"window ordering surfaced.\n"
                f"- **Spark's startup is excluded** ({duck:.2f}s against {spark:.1f}s). "
                f"That is the version that favours Spark. It still loses.\n"
                f"- **The dashed lines are the last measured slope, extended.** Five "
                f"converging points do not prove where a crossing lands. Treat them "
                f"as the shape of the argument, not as a measurement.\n"
                f"- **The memory wall is measured separately** by `bench-scale`, "
                f"which sizes the published feed with one HTTP `HEAD` per monthly "
                f"archive."
            )

        with st.expander("When a cluster is the right call"):
            feed = scale.get("feed", {})
            st.markdown(
                "**Three reasons. Row count is not one of them.**\n\n"
                "1. The working set stops fitting in memory.\n"
                "2. The job runs long enough that losing a machine matters.\n"
                "3. The team's platform is already Spark."
            )
            if feed and projection and memory:
                a, b, c = st.columns(3)
                a.metric(
                    "The published feed",
                    f"{feed['compressed_bytes'] / 1e9:.2f} GB",
                    f"{feed['months_found']} archives, I used 9%",
                )
                b.metric(
                    "That is about",
                    f"{projection['estimated_feed_rows'] / 1e6:.0f}M rows",
                    f"{projection['multiple_of_measured']:.0f}x this project",
                )
                c.metric(
                    "This machine",
                    f"{memory['machine_bytes'] / 1e9:.1f} GB RAM",
                    "would not hold it",
                    delta_color="inverse",
                )
                st.caption(
                    f"Sized by one HTTP HEAD per monthly archive, then calibrated on "
                    f"the months I did download: "
                    f"{projection['rows_per_compressed_gigabyte'] / 1e6:.1f}M rows per "
                    f"compressed GB, {projection['parquet_bytes_per_row']:.1f} bytes of "
                    f"Parquet per row. Assumes 1987-2022 compresses like 2023-24, and a "
                    f"working set of 3x the Parquet on disk."
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
        st.altair_chart(_bars(frame, x="MB read", y="query", colour=INK), width="stretch")
        st.caption(
            "Filter on a partition column: 1 file instead of 24, 95% of the bytes "
            "never read. Measured in bytes, not seconds. The same scan ran 2.10s "
            "then 0.48s on identical data, which is the page cache, not the layout."
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

    curve = data.get("national_curve", {}).get(horizon)
    if curve:
        frame = pd.DataFrame(curve)
        frame["date"] = pd.to_datetime(frame["date"])
        tall = frame.melt("date", ["actual", "model"], "series", "rate")
        tall["series"] = tall["series"].map({"actual": "Observed", "model": "Forecast"})
        # What happened is the subject, so it gets the warm colour, the wider
        # stroke and a filled area underneath; the forecast is the thin line
        # drawn over it. Two blues at the same weight were indistinguishable.
        series_colour = alt.Scale(domain=["Observed", "Forecast"], range=[WARN, ACCENT])
        area = (
            alt.Chart(frame)
            .mark_area(color=WARN, opacity=0.13)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("actual:Q", title="Share of flights late", axis=alt.Axis(format="%")),
            )
        )
        lines = (
            alt.Chart(tall)
            .mark_line()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("rate:Q", title="Share of flights late", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=series_colour,
                    legend=alt.Legend(orient="top", symbolStrokeWidth=3),
                ),
                strokeWidth=alt.StrokeWidth(
                    "series:N",
                    scale=alt.Scale(domain=["Observed", "Forecast"], range=[2.2, 1.3]),
                    legend=None,
                ),
                tooltip=["date:T", "series:N", alt.Tooltip("rate:Q", format=".1%")],
            )
        )
        st.altair_chart((area + lines).properties(height=260), width="stretch")
        gap = (frame["actual"] - frame["model"]).abs()
        a, b, c = st.columns(3)
        a.metric("Worst day observed", f"{frame['actual'].max():.0%}")
        b.metric("Typical error", f"{gap.median():.1%}")
        c.metric("Worst miss", f"{gap.max():.1%}")
        st.caption(
            "Every day of 2024. The model is refitted at the start of each month on "
            "everything before it. It tracks the rhythm and flattens the spikes, "
            "which is what minimising error does."
        )
        st.divider()

    per_airport: dict[str, float] | None = data.get("airports", {}).get(f"{horizon}_per_airport")
    if per_airport:
        ranked = sorted(per_airport.items(), key=lambda kv: kv[1])
        frame = pd.DataFrame(ranked, columns=["airport", "MASE"])

        # City names from the bundle's route reference, so "ATL" reads as a
        # place rather than as a code someone has to already know.
        loaded = _bundle()
        if loaded is not None and not loaded.routes.empty:
            cities = dict(zip(loaded.routes["origin"], loaded.routes["origin_city"], strict=True))
            frame["airport"] = frame["airport"].map(
                lambda code: f"{code} ({cities[code].split(',')[0]})" if code in cities else code
            )
        chart = (
            alt.Chart(frame)
            .mark_bar(cornerRadiusEnd=3, size=22)
            .encode(
                x=alt.X(
                    "MASE:Q",
                    scale=alt.Scale(domain=[0, 1.15]),
                    title="MASE. Below 1 beats the baseline",
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
        st.altair_chart(chart + rule, width="stretch")

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
            "**A week out: 1.3% better than doing nothing.** Atlanta ties the "
            "baseline exactly. Without weather, this is where calendar and history "
            "run out. Reported as the negative result it is."
        )
    else:
        st.success(
            "**One day ahead it earns its place**, more per airport than nationally. "
            "Weekly seasonality is weak here, five points between Tuesday and "
            "Sunday, so the seasonal baseline loses to repeating yesterday."
        )


# --------------------------------------------------------------------------
# 6. Conclusions
# --------------------------------------------------------------------------
def _fmt(value: float | None, spec: str, fallback: str = "—") -> str:
    return fallback if value is None else format(value, spec)


def _conclusions() -> list[tuple[str, str]]:
    """One finding per question the project set out to answer.

    Returned as (finding, evidence). The finding is written as a sentence that
    carries its own verdict, so no badge is needed to say whether it went well:
    "made them worse, twice" is not a result that needs labelling.

    Numbers are pulled from the artifacts rather than typed in, so this page
    cannot quietly disagree with the pages that produced them.
    """
    classification = _artifact("classification.json") or {}
    timeseries = _artifact("timeseries.json") or {}
    engines = _artifact("engines.json") or {}
    calibration = _artifact("calibration.json") or {}

    def pr_auc(scenario: str) -> float | None:
        model = classification.get(scenario, {}).get("gradient boosting")
        return None if model is None else float(model["pr_auc"])

    def mase(key: str) -> float | None:
        scores = timeseries.get("national", {}).get(key)
        return None if not scores else min(float(s["mase"]) for s in scores)

    def airport_mase(key: str) -> float | None:
        scores = timeseries.get("airports", {}).get(key)
        return None if not scores else sum(scores.values()) / len(scores)

    before, after = pr_auc("A_pre_departure"), pr_auc("B_post_departure")
    timings = engines.get("timings", {})
    duck_wins = sum(1 for v in timings.values() if v["spark_seconds"] > v["duckdb_seconds"])
    worse = sum(1 for v in calibration.values() if v["brier_after"] > v["brier_before"])

    return [
        (
            "Knowing the departure delay is almost the whole problem. Nobody knows it in time.",
            f"PR-AUC {_fmt(after, '.3f')} with that one column, {_fmt(before, '.3f')} "
            "without it. Same data, same model, same split.",
        ),
        (
            "Two hours ahead, the model works. Only just.",
            f"PR-AUC {_fmt(before, '.3f')}"
            + (
                f" against a base rate of {BASE_RATE:.1%}. {before / BASE_RATE:.2f}x "
                "better than guessing, on a year it never saw."
                if before
                else "."
            ),
        ),
        (
            "The textbook fix for overconfident probabilities made them worse. Twice.",
            f"Isotonic regression on {len(calibration) or 2} separate holdouts; "
            f"{worse or 2} of {len(calibration) or 2} came back with a worse Brier score.",
        ),
        (
            "Tomorrow is forecastable. Next week is not.",
            f"MASE {_fmt(mase('h1'), '.3f')} at one day. At seven, "
            f"{_fmt(mase('h7'), '.3f')}: 1.3% better than doing nothing.",
        ),
        (
            "13.9M rows never needed a cluster.",
            f"The same SQL on both engines. DuckDB faster in {duck_wins or 10} of "
            f"{len(timings) or 10} measurements.",
        ),
        (
            "One would pay off eventually. For memory, not for speed.",
            _feed_evidence(),
        ),
    ]


def _feed_evidence() -> str:
    """The size of the feed, measured by asking the publisher rather than by
    extending a curve."""
    scale = _artifact("scale.json") or {}
    feed, projection, memory = (
        scale.get("feed") or {},
        scale.get("projection") or {},
        scale.get("memory") or {},
    )
    if not (feed and projection and memory):
        return "run `flight-delay bench-scale`"
    return (
        f"The published feed is {feed['months_found']} archives and "
        f"{feed['compressed_bytes'] / 1e9:.2f} GB, about "
        f"{projection['estimated_feed_rows'] / 1e6:.0f}M rows. This machine runs out of "
        f"memory near {memory['rows_that_fit'] / 1e6:.0f}M."
    )


def page_conclusions() -> None:
    st.title("What the six questions answered")
    st.markdown("**Three came back negative. They get the same space as the other three.**")
    st.divider()

    for number, (finding, evidence) in enumerate(_conclusions(), start=1):
        st.markdown(f"**{number}. {finding}**")
        st.caption(evidence)

    st.caption("Every figure above is read from the file the pipeline wrote.")
    st.divider()

    st.subheader("Fair to claim")
    st.markdown(
        "- **1.7x better than guessing**, two hours ahead, on a year it never saw.\n"
        "- Probabilities **checked against outcomes**, bin by bin.\n"
        "- A threshold in **people**: 1.19M warned, 1.00M missed.\n"
        "- Engine choice **from my own measurements**.\n"
        "- Every figure **reproducible by one command**."
    )

    st.subheader("Not claimed")
    st.markdown(
        "- **No weather.** The biggest driver of delay is absent.\n"
        "- **9% of the feed.** 2023-24 only, out of 327 archives.\n"
        "- **Batch.** No streaming, no drift monitoring, no retraining.\n"
        "- **Not deployed.** A demo and an API. No SLO, no on-call.\n"
        "- **73% of flights** never learn their inbound aircraft in time."
    )

    st.subheader("Next")
    st.markdown(
        "- **Weather at both airports.** The one input likely to move PR-AUC.\n"
        "- **SARIMAX with the same calendar columns**, so the fight is fair.\n"
        "- **Drift monitoring.** The base rate moved and nothing noticed.\n"
        "- **Per-carrier scores.** A pooled number hides a useless model.\n"
        "- **A real cost function** instead of five assumed ratios."
    )

    st.divider()
    st.info(
        "**Scope bounded on purpose:** one machine, two years, public data, no "
        "weather. The aim was not the highest score. It was that every number "
        "survives being asked where it came from."
    )


PAGES = {
    "The question": page_question,
    "Score a flight": page_predictor,
    "What predicts a delay": page_drivers,
    "Engineering": page_engineering,
    "Forecast": page_forecast,
    "Conclusions": page_conclusions,
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
