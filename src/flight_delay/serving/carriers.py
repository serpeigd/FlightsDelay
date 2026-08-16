"""Airline names for display.

The BTS feed carries only the IATA code in ``Reporting_Airline``; there is no
name column anywhere in it. This mapping is therefore **hardcoded and
display-only** — it never reaches the model, which sees the code exactly as the
feed reports it.

It covers the fifteen carriers that appear in 2023-2024. Several are regional
operators flying under a major's brand: Endeavor and Envoy feed Delta and
American respectively, SkyWest and Republic fly for several. The code is what
the data reports, so the code stays first.
"""

from __future__ import annotations

from typing import Final

NAMES: Final[dict[str, str]] = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "NK": "Spirit Airlines",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "YX": "Republic Airways",
}


def label(code: str) -> str:
    """``'AA'`` to ``'AA (American Airlines)'``.

    An unknown code returns unchanged rather than raising: a carrier that starts
    reporting next year should not break a dropdown.
    """
    name = NAMES.get(code.upper())
    return f"{code} ({name})" if name else code
