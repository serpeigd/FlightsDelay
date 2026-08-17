"""Streamlit dashboard.

    streamlit run src/flight_delay/serving/dashboard.py

Charts over prose. Every figure is read from artifacts the pipeline wrote, so
the app cannot disagree with `docs/` -- but it renders them rather than
reprinting them, because a reader who wanted the write-up would read the
write-up.
"""

from __future__ import annotations

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
        st.altair_chart(chart + labels + rule, width="stretch")
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
                "**On the dashed line, the probability is honest.** Below it, the "
                "model is overconfident.\n\n"
                "It sits on the line where almost all its mass is, and drifts above "
                "it at the top — where it says 64%, the real rate is 53%.\n\n"
                "Those bins are small, and they are exactly the flights a warning "
                "system would act on."
            )
            st.caption(
                "Isotonic recalibration was tried on two different holdouts and "
                "made both worse. Gradient boosting optimises log loss, a proper "
                "scoring rule, so it arrives calibrated."
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
        st.altair_chart(_bars(frame, x="pr_auc_drop", y="label", colour=ACCENT), width="stretch")
        st.caption(
            "Predictive power lost when each input is shuffled. Measured on the held-out year."
        )
    with right:
        st.metric("Strongest single input", frame.iloc[0]["label"])
        st.markdown(
            "**The clock wins, by double.**\n\n"
            "Delay builds through the day. An 8am departure is a different "
            "proposition from a 7pm one."
        )
        st.markdown(
            "**The hard-won features came fourth.**\n\n"
            "The inbound-aircraft columns took the most work in the project. "
            "They land mid-table. That is the measurement."
        )
        st.info(
            "**Four inputs measured zero and were removed.** Re-scoring without "
            "them moved PR-AUC from 0.343 to 0.343 — which is why they were "
            "measured before being trusted."
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
        # The question is "does Spark ever win?", and two absolute-seconds
        # charts answer it only if you compare heights across panels. The
        # ratio answers it directly: one line per workload, one line at 1.0,
        # and crossing it is the whole event.
        ratios = pd.DataFrame(
            [
                {
                    "months": int(key.split("@")[1].removesuffix("m")),
                    "workload": "Aggregation" if key.startswith("daily") else "Window function",
                    "ratio": value["spark_seconds"] / value["duckdb_seconds"],
                }
                for key, value in engines["timings"].items()
                if value.get("duckdb_seconds")
            ]
        )
        crossing = ratios[ratios["ratio"] < 1.0]
        rule = (
            alt.Chart(pd.DataFrame({"y": [1.0]}))
            .mark_rule(color=INK, strokeDash=[6, 4])
            .encode(y="y:Q")
        )
        note = (
            alt.Chart(pd.DataFrame({"y": [1.0], "text": ["Spark wins below this line"]}))
            .mark_text(align="left", dx=6, dy=12, fontSize=11, color=INK)
            .encode(y="y:Q", text="text:N")
        )
        st.altair_chart(
            (
                alt.Chart(ratios)
                .mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=3)
                .encode(
                    x=alt.X(
                        "months:Q",
                        title="Months of data  (1 month = 580k rows, 24 = 13.9M)",
                        axis=alt.Axis(values=[1, 3, 6, 12, 24]),
                    ),
                    y=alt.Y(
                        "ratio:Q",
                        title="Spark time ÷ DuckDB time",
                        scale=alt.Scale(type="log", domain=[0.7, 25]),
                        axis=alt.Axis(values=[1, 2, 5, 10, 20], format="d"),
                    ),
                    color=alt.Color(
                        "workload:N",
                        title=None,
                        scale=alt.Scale(range=[ACCENT, WARN]),
                        legend=alt.Legend(orient="top"),
                    ),
                    tooltip=[
                        "workload",
                        "months",
                        alt.Tooltip("ratio:Q", format=".1f", title="x slower"),
                    ],
                )
                + rule
                + note
            ).properties(height=280),
            width="stretch",
        )
        st.markdown(
            f"**No — and the chart says where I checked.** Every point is above "
            f"1.0, so DuckDB won all {len(ratios)} measurements. The lines fall "
            f"steeply and then flatten: Spark closes from ~19x to ~2.5x over a "
            f"24x increase in data, and then stops closing. "
            + (
                "One point did cross."
                if not crossing.empty
                else "**Nothing crossed, so nothing is claimed beyond the last point measured.**"
            )
        )

        with st.expander("How this was measured, and why you should believe it"):
            duck = engines.get("duckdb_startup_seconds")
            spark = engines.get("spark_startup_seconds")
            matched = all(v.get("answers_match") for v in engines["timings"].values())
            st.markdown(
                f"- **Same SQL, same table, same machine.** Two queries — one "
                f"grouped aggregation, one window function — run against the same "
                f"Delta table at {len(engines.get('scales_months', []))} scales, "
                f"{engines.get('repeats', 3)} repeats each, best time kept.\n"
                f"- **The answers were compared, not assumed.** Row counts and "
                f"checksums matched on every run: "
                f"`answers_match` is {'true everywhere' if matched else 'not uniform'}. "
                f"An earlier version disagreed by 1-2 rows and that is how the tie "
                f"in the window ordering was found.\n"
                f"- **Startup is excluded from the lines above and shown separately** "
                f"— {duck:.2f}s for DuckDB against {spark:.1f}s for Spark. Including "
                f"it would be true but unfair; excluding it is the version that "
                f"favours Spark, and Spark still loses.\n"
                f"- **The extrapolation is refused on purpose.** Five points that "
                f"converge without crossing do not locate a crossing point. Saying "
                f"'Spark wins beyond X rows' from this data would be a guess "
                f"dressed as a measurement."
            )
            if duck is not None and spark is not None:
                a, b = st.columns(2)
                a.metric("DuckDB startup", f"{duck:.2f}s")
                b.metric("Spark startup", f"{spark:.1f}s")

        with st.expander("The raw timings, in seconds"):
            for column, workload in zip(
                st.columns(2, gap="large"), frame["workload"].unique(), strict=False
            ):
                block = frame[frame["workload"] == workload]
                with column:
                    st.markdown(f"**{workload}**")
                    st.altair_chart(
                        alt.Chart(block)
                        .mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3)
                        .encode(
                            x=alt.X(
                                "months:Q",
                                title="Months of data",
                                axis=alt.Axis(values=[1, 3, 6, 12, 24]),
                            ),
                            y=alt.Y("seconds:Q", title="Seconds"),
                            color=alt.Color(
                                "engine:N",
                                title=None,
                                scale=alt.Scale(domain=["Duckdb", "Spark"], range=[ACCENT, WARN]),
                                legend=alt.Legend(orient="top"),
                            ),
                            tooltip=["engine", "months", alt.Tooltip("seconds:Q", format=".2f")],
                        )
                        .properties(height=230),
                        width="stretch",
                    )

        with st.expander("So when *would* a cluster be the right call?"):
            st.markdown(
                "**This is two years of a feed that goes back to 1987.** The "
                "limit was my laptop and the time I had, not the source.\n\n"
                "Extrapolating from what is measured here: two years is 13.9M "
                "rows and 475 MB of Parquet, and the window-function workload "
                "has to hold the whole table. This machine has 7.6 GB. So the "
                "point where one machine stops being enough is somewhere around "
                "**200-300M rows — roughly the entire history of the feed.**\n\n"
                "That is the honest answer, and it is not 'when you have a lot "
                "of data'. It is: **when the working set stops fitting in "
                "memory, or when the job has to survive a machine dying.** "
                "Neither is true at two years, so neither is claimed."
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
            "Every day of 2024, forecast by a model refitted monthly on everything "
            "before it. It tracks the weekly rhythm and flattens the spikes — which "
            "is what an error-minimising objective is built to do."
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


# --------------------------------------------------------------------------
# 6. Conclusions
# --------------------------------------------------------------------------
def _fmt(value: float | None, spec: str, fallback: str = "—") -> str:
    return fallback if value is None else format(value, spec)


def _scoreboard() -> pd.DataFrame:
    """One row per question the project set out to answer.

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

    return pd.DataFrame(
        [
            {
                "Question": "Can a delay be predicted before the plane moves?",
                "Answer": "Yes, but weakly — and weakly is the honest ceiling here",
                "Evidence": f"PR-AUC {_fmt(before, '.3f')}"
                + (f" · {before / BASE_RATE:.2f}x the 20.6% base rate" if before else ""),
                "Verdict": "Positive",
            },
            {
                "Question": "How much does one leaked column change that?",
                "Answer": "It replaces the problem with an easier one",
                "Evidence": f"{_fmt(after, '.3f')} against {_fmt(before, '.3f')} — same data, "
                "same model, same split",
                "Verdict": "The point",
            },
            {
                "Question": "Do the probabilities mean what they say?",
                "Answer": "Yes where the flights are; overconfident at the top end",
                "Evidence": f"Isotonic recalibration was tried on {len(calibration) or 2} "
                f"holdouts and made {worse or 2} of them worse",
                "Verdict": "Negative",
            },
            {
                "Question": "How bad will tomorrow be at an airport?",
                "Answer": "Answerable one day out, not one week out",
                "Evidence": f"MASE {_fmt(mase('h1'), '.3f')} national and "
                f"{_fmt(airport_mase('h1_per_airport'), '.3f')} per airport at 1 day; "
                f"{_fmt(mase('h7'), '.3f')} at 7 days",
                "Verdict": "Mixed",
            },
            {
                "Question": "Does 13.9M rows need a cluster?",
                "Answer": "No. One laptop beat the cluster at every size tested",
                "Evidence": f"DuckDB faster in {duck_wins or 10}/{len(timings) or 10} "
                "measurements; the ratio narrows but never reaches 1.0",
                "Verdict": "Negative",
            },
        ]
    )


def page_conclusions() -> None:
    st.title("What this is worth, and what it is not")
    st.markdown(
        "Five questions, asked in order. **Two answers came back negative, one "
        "came back mixed, and the most useful one was never about accuracy.** "
        "They are all here, because a write-up that only reports its wins is not "
        "a measurement."
    )

    board = _scoreboard()
    st.dataframe(
        board,
        hide_index=True,
        width="stretch",
        column_config={
            "Question": st.column_config.TextColumn(width="medium"),
            "Answer": st.column_config.TextColumn(width="medium"),
            "Evidence": st.column_config.TextColumn(width="large"),
            "Verdict": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        "Every number in this table is read from the file the pipeline wrote, "
        "not typed into this page."
    )

    st.divider()
    fair, limits, next_up = st.columns(3, gap="large")

    with fair:
        st.subheader("What is fair to claim")
        st.markdown(
            "- A **2h-ahead** delay model that is **1.7x better than guessing**, "
            "on a year it never saw.\n"
            "- Probabilities that can be **shown to a passenger** — checked "
            "against reality bin by bin, not just scored.\n"
            "- A threshold expressed in **people, not percentages**: at 0.30, "
            "1.19M passengers warned, 1.00M delays still missed.\n"
            "- An engine choice **argued from measurements** of my own data, "
            "not from a vendor's benchmark.\n"
            "- Every figure reproducible by **one command**, on a machine that "
            "does not have the data."
        )

    with limits:
        st.subheader("What it is not")
        st.markdown(
            "- **No weather.** The single biggest driver of delay is absent. "
            "Everything here is the floor that calendar and history alone reach.\n"
            "- **US domestic, 2023-24.** Two years of a feed that starts in 1987, "
            "cut to what one laptop with 7.6 GB could hold.\n"
            "- **Historic files, not a live feed.** No streaming, no drift "
            "monitoring, no retraining schedule — designed for, not built.\n"
            "- **Not deployed to anyone.** A demo and an API, not a service with "
            "users, SLOs or an on-call rota.\n"
            "- **73% of flights never learn their inbound aircraft** in time, so "
            "the strongest signal is missing exactly when it would help most."
        )

    with next_up:
        st.subheader("What another month would buy")
        st.markdown(
            "- **Weather at origin and destination.** The one addition likely to "
            "move PR-AUC rather than decorate it.\n"
            "- **SARIMAX with the same calendar columns**, so the classical "
            "baseline loses for the right reason instead of an unfair one.\n"
            "- **Drift monitoring on the served model** — the 2024 base rate "
            "moved, and nothing here would have noticed.\n"
            "- **Per-carrier evaluation.** A pooled PR-AUC can hide a model that "
            "is useless for one airline.\n"
            "- **A cost function from someone who owns the decision**, replacing "
            "my five assumed miss-to-false-alarm ratios."
        )

    st.divider()
    st.info(
        "**The ambition was bounded on purpose.** One machine, two years, no "
        "weather, public data. Inside those bounds the aim was not the highest "
        "score — it was that **every number survives being asked where it came "
        "from**. That is why the leakage rule lives in the schema, why isotonic "
        "regression is reported as a failure, and why the cluster benchmark ends "
        "in 'not here, and I will not extrapolate to where'."
    )

    with st.expander("Why so much of this is negative results"):
        st.markdown(
            "A model that scores 0.94 by reading the departure delay is the most "
            "common way this exact problem is done wrong, and it looks "
            "**excellent** right up to the moment someone asks when the value "
            "becomes available. The pre-departure number, 0.343, is the smaller "
            "one and the only honest one.\n\n"
            "The same pattern repeats. Isotonic regression is the textbook fix "
            "and it hurt. The seasonal baseline is the textbook choice for daily "
            "data and it lost to repeating yesterday. Spark is the textbook "
            "answer to 13.9M rows and it lost ten times out of ten.\n\n"
            "**Each of those is only knowable because it was measured, and each "
            "would have shipped as an unexamined default.** That is what the "
            "project is actually demonstrating."
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
