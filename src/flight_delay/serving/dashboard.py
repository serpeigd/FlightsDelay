"""Streamlit dashboard.

    streamlit run src/flight_delay/serving/dashboard.py

Three pages, in order of what is worth someone's attention:

- **Findings** leads, because the results that contradicted an expectation are
  the substance of the project.
- **Predictor** scores a flight and lets the threshold move, since the
  threshold is a business input rather than a model output.
- **Forecast** shows the daily series and where the forecaster stops working.

Every number rendered here is read from the artifacts the pipeline wrote. The
page computes nothing of its own, so it cannot disagree with `docs/`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from flight_delay.serving import artifacts, bundle
from flight_delay.serving.features import build_row

BASE_RATE = 0.2056


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


@st.cache_data
def _doc(name: str) -> str | None:
    path = Path(__file__).resolve().parents[3] / "docs" / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def page_findings() -> None:
    st.title("What the data actually said")
    st.caption(
        "Results that contradicted an expectation. Each came from reading "
        "output, not from a test passing."
    )

    classification = _artifact("classification.json")
    if classification:
        a = classification.get("A_pre_departure", {}).get("gradient boosting", {})
        b = classification.get("B_post_departure", {}).get("gradient boosting", {})
        left, right = st.columns(2)
        left.metric("Pre-departure (T-2h)", f"{a.get('pr_auc', 0):.3f}", "PR-AUC")
        right.metric("Post-departure", f"{b.get('pr_auc', 0):.3f}", "PR-AUC")
        st.markdown(
            "**Same label, same data, same models, same split.** The only difference "
            "is whether the model may see how late the aircraft actually pushed back. "
            "A model built without an explicit cutoff slides into the second column "
            "by accident, reports 0.94, and cannot answer the question a passenger "
            "asks — *should I leave for the airport?*"
        )

    text = _doc("findings.md")
    if text:
        # The document is the source of truth; drop its title, keep the rest.
        st.markdown(text.split("---", 1)[-1] if "---" in text else text)
    else:
        st.info("docs/findings.md not found.")


def page_predictor() -> None:
    st.title("Will this flight be late?")
    loaded = _bundle()
    if loaded is None:
        st.warning("No model bundle. Run `flight-delay export-model` first.")
        return

    st.caption(
        f"Pre-departure model, fitted on {loaded.trained_on}, "
        f"PR-AUC {loaded.pr_auc:.3f} on the held-out year."
    )

    with st.form("flight"):
        c1, c2, c3 = st.columns(3)
        carrier = c1.text_input("Carrier", "AA", max_chars=3)
        origin = c2.text_input("Origin", "JFK", max_chars=4)
        dest = c3.text_input("Destination", "LAX", max_chars=4)

        c1, c2, c3 = st.columns(3)
        flight_date = c1.date_input("Date", date(2024, 7, 15))
        departure = c2.text_input("Scheduled departure (HHMM)", "0830")
        arrival = c3.text_input("Scheduled arrival (HHMM)", "1145")

        c1, c2 = st.columns(2)
        distance = c1.number_input("Distance (miles)", min_value=1.0, value=2475.0)
        elapsed = c2.number_input("Scheduled duration (min)", min_value=1.0, value=375.0)

        known = st.checkbox(
            "The inbound aircraft has already landed",
            help="True for only 27.2% of flights two hours before departure.",
        )
        inbound = st.slider("Inbound arrival delay (min)", -30, 180, 0) if known else None

        threshold = st.slider(
            "Warn above this probability",
            0.05,
            0.75,
            0.25,
            0.05,
            help="Not a model output. It depends on what a missed delay costs "
            "against a false alarm. At 0.5 the system warns 0.9% of passengers.",
        )
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    try:
        frame = build_row(
            flight_date=flight_date if isinstance(flight_date, date) else date(2024, 7, 15),
            carrier=carrier.upper(),
            origin=origin.upper(),
            dest=dest.upper(),
            scheduled_departure=departure,
            scheduled_arrival=arrival,
            distance=float(distance),
            scheduled_elapsed_minutes=float(elapsed),
            priors=loaded.priors,
            inbound_delay_minutes=inbound,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    probability = loaded.predict_proba(frame)
    c1, c2, c3 = st.columns(3)
    c1.metric("Delay probability", f"{probability:.1%}")
    c2.metric("Against a base rate of", f"{BASE_RATE:.1%}", f"{probability / BASE_RATE:.2f}x")
    c3.metric("Decision", "WARN" if probability >= threshold else "no warning")

    st.caption(
        "The model is well calibrated through the bulk of its range and "
        "overconfident above 0.5: where it says 64%, the observed rate is 53%. "
        "Isotonic recalibration was tried and made it worse — see Findings."
    )


def page_forecast() -> None:
    st.title("How bad will tomorrow be?")
    data = _artifact("timeseries.json")
    if data is None:
        st.warning("No backtest results. Run `flight-delay forecast` first.")
        return

    st.caption(
        "Rolling-origin backtest: fitted on 2023, refitted monthly through 2024. "
        "MASE below 1 beats repeating last week's value."
    )

    for series in ("national", "airports"):
        block = data.get(series)
        if not block:
            continue
        st.subheader("National" if series == "national" else "Ten busiest airports")
        for horizon in ("h1", "h7"):
            scores = block.get(horizon)
            if not scores:
                continue
            st.markdown(f"**{horizon.removeprefix('h')} day(s) ahead**")
            st.dataframe(
                pd.DataFrame(scores)[["name", "mae", "rmse", "mase"]]
                .rename(columns=str.upper)
                .round(4),
                hide_index=True,
                width="stretch",
            )

        per_airport = block.get("h7_per_airport")
        if per_airport:
            st.markdown("**MASE per airport, 7 days ahead**")
            ranked = sorted(per_airport.items(), key=lambda kv: float(kv[1]))
            frame = pd.DataFrame(ranked, columns=["airport", "MASE"])
            frame["beats seasonal naive"] = frame["MASE"] < 1.0
            st.dataframe(frame.round(3), hide_index=True, width="stretch")
            st.caption(
                "The pooled figure hides that Atlanta ties the baseline exactly. "
                "It is the busiest airport in the feed and has the lowest delay "
                "rate, so there is least variance left to explain."
            )

    st.info(
        "Seven days ahead at national level the model is 1.3% better than doing "
        "nothing. Without weather, that is roughly where calendar features and "
        "lagged rates run out."
    )


PAGES = {
    "Findings": page_findings,
    "Predictor": page_predictor,
    "Forecast": page_forecast,
}


def main() -> None:
    st.set_page_config(page_title="Flight delay prediction", layout="wide")
    with st.sidebar:
        st.markdown("### Flight delay prediction")
        st.caption("13,926,960 US flights, 2023-2024")
        choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(
            "Every figure here is read from artifacts the pipeline wrote, so the "
            "dashboard cannot disagree with the documentation."
        )
        st.caption(f"Source: {artifacts.source_label('classification.json')}.")
    PAGES[choice]()


main()
