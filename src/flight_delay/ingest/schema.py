"""The column contract for BTS On-Time Performance data.

This module answers one question for every column: **when does this value
become known?** That is the difference between a model worth deploying and a
model that reads the answer off the back of the card.

The raw feed carries 109 named columns (plus a phantom empty field, because
every header and data line ends with a trailing comma). Roughly half of them
describe diverted-flight legs (``Div1*``-``Div5*``) and are almost entirely
null; they are dropped. Of what remains, the columns that matter most for
predicting ``ArrDel15`` -- ``DepDelay``, ``TaxiOut``, and the
``CarrierDelay``/``WeatherDelay``/``NASDelay``/``LateAircraftDelay``
breakdown -- do not exist until after the flight has departed or landed.

``LateAircraftDelay`` is the clearest trap: it is an after-the-fact accounting
of *why* the flight was late, published together with the delay itself.

Encoding availability in the schema, rather than in a comment or a notebook
cell, means a feature set is assembled by filtering on it and leakage becomes
a type error rather than a suspiciously good score.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal

SqlType = Literal["string", "int", "double", "date"]


class Availability(Enum):
    """When a column's value becomes known, relative to the flight."""

    #: In the published schedule, hours or days ahead. The only inputs allowed
    #: in scenario A (pre-departure).
    SCHEDULED = "scheduled"
    #: Known once the aircraft actually pushes back and lifts off. Allowed in
    #: scenario B (post-departure) only.
    DEPARTED = "departed"
    #: Known at or after touchdown. Never a feature -- these are the outcome,
    #: or a post-hoc explanation of it.
    ARRIVED = "arrived"
    #: The target itself.
    LABEL = "label"


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    sql_type: SqlType
    availability: Availability
    note: str = ""


#: Clock fields arrive zero-padded ("0800", "0757") and can read "2400" for
#: midnight, so they are kept as strings and converted deliberately rather than
#: being silently coerced to integers.
_CLOCK: Final[SqlType] = "string"


COLUMNS: Final[tuple[Column, ...]] = (
    # --- calendar -----------------------------------------------------------
    Column("Year", "int", Availability.SCHEDULED),
    Column("Quarter", "int", Availability.SCHEDULED),
    Column("Month", "int", Availability.SCHEDULED),
    Column("DayofMonth", "int", Availability.SCHEDULED),
    Column("DayOfWeek", "int", Availability.SCHEDULED),
    Column("FlightDate", "date", Availability.SCHEDULED),
    # --- carrier and aircraft ----------------------------------------------
    Column("Reporting_Airline", "string", Availability.SCHEDULED),
    Column("DOT_ID_Reporting_Airline", "int", Availability.SCHEDULED),
    Column("Flight_Number_Reporting_Airline", "string", Availability.SCHEDULED),
    Column(
        "Tail_Number",
        "string",
        Availability.SCHEDULED,
        note=(
            "BTS records the aircraft that actually operated the leg. The "
            "planned rotation is normally known hours ahead, so this is "
            "treated as scheduled, but a same-day aircraft swap would not "
            "have been visible at T-2h. Used only to derive the inbound-leg "
            "feature, never as an identifier the model can memorise."
        ),
    ),
    # --- origin and destination --------------------------------------------
    Column("OriginAirportID", "int", Availability.SCHEDULED),
    Column("Origin", "string", Availability.SCHEDULED),
    Column("OriginCityName", "string", Availability.SCHEDULED),
    Column("OriginState", "string", Availability.SCHEDULED),
    Column("DestAirportID", "int", Availability.SCHEDULED),
    Column("Dest", "string", Availability.SCHEDULED),
    Column("DestCityName", "string", Availability.SCHEDULED),
    Column("DestState", "string", Availability.SCHEDULED),
    # --- planned times ------------------------------------------------------
    Column("CRSDepTime", _CLOCK, Availability.SCHEDULED),
    Column("DepTimeBlk", "string", Availability.SCHEDULED),
    Column("CRSArrTime", _CLOCK, Availability.SCHEDULED),
    Column("ArrTimeBlk", "string", Availability.SCHEDULED),
    Column("CRSElapsedTime", "double", Availability.SCHEDULED),
    Column("Distance", "double", Availability.SCHEDULED),
    Column("DistanceGroup", "int", Availability.SCHEDULED),
    Column("Flights", "double", Availability.SCHEDULED),
    # --- observed at departure ---------------------------------------------
    Column("DepTime", _CLOCK, Availability.DEPARTED),
    Column("DepDelay", "double", Availability.DEPARTED),
    Column("DepDelayMinutes", "double", Availability.DEPARTED),
    Column("DepDel15", "double", Availability.DEPARTED),
    Column("DepartureDelayGroups", "double", Availability.DEPARTED),
    Column("TaxiOut", "double", Availability.DEPARTED),
    Column("WheelsOff", _CLOCK, Availability.DEPARTED),
    # --- observed at arrival: outcome, never an input -----------------------
    Column("WheelsOn", _CLOCK, Availability.ARRIVED),
    Column("TaxiIn", "double", Availability.ARRIVED),
    Column("ArrTime", _CLOCK, Availability.ARRIVED),
    Column("ArrDelay", "double", Availability.ARRIVED),
    Column("ArrDelayMinutes", "double", Availability.ARRIVED),
    Column("ArrivalDelayGroups", "double", Availability.ARRIVED),
    Column("ActualElapsedTime", "double", Availability.ARRIVED),
    Column("AirTime", "double", Availability.ARRIVED),
    Column("Cancelled", "double", Availability.ARRIVED),
    Column("CancellationCode", "string", Availability.ARRIVED),
    Column("Diverted", "double", Availability.ARRIVED),
    Column(
        "CarrierDelay",
        "double",
        Availability.ARRIVED,
        note="Post-hoc attribution of the delay being predicted.",
    ),
    Column("WeatherDelay", "double", Availability.ARRIVED),
    Column("NASDelay", "double", Availability.ARRIVED),
    Column("SecurityDelay", "double", Availability.ARRIVED),
    Column(
        "LateAircraftDelay",
        "double",
        Availability.ARRIVED,
        note="Knock-on delay from the inbound leg, attributed after landing.",
    ),
    # --- target -------------------------------------------------------------
    Column(
        "ArrDel15",
        "double",
        Availability.LABEL,
        note="Null for cancelled and diverted flights (2.2% of rows).",
    ),
)

BY_NAME: Final[dict[str, Column]] = {c.name: c for c in COLUMNS}

#: Columns the ingestion step reads out of the raw CSV.
KEPT_COLUMNS: Final[tuple[str, ...]] = tuple(c.name for c in COLUMNS)

#: Partition keys of the curated Delta table. Year/month matches how the data
#: arrives and how it is queried, which is what makes partition pruning pay.
PARTITION_BY: Final[tuple[str, ...]] = ("Year", "Month")


def columns_available_at(*availability: Availability) -> tuple[str, ...]:
    """Column names knowable at the given points in a flight's life.

    Feature builders call this instead of listing columns by hand, so adding a
    column to the schema cannot accidentally add it to a feature set.
    """
    wanted = set(availability)
    return tuple(c.name for c in COLUMNS if c.availability in wanted)


#: Scenario A: everything in the published schedule, nothing else.
SCENARIO_A_COLUMNS: Final[tuple[str, ...]] = columns_available_at(Availability.SCHEDULED)

#: Scenario B: the aircraft is airborne, so the departure record is fair game.
SCENARIO_B_COLUMNS: Final[tuple[str, ...]] = columns_available_at(
    Availability.SCHEDULED, Availability.DEPARTED
)
