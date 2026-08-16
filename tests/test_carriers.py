from __future__ import annotations

import pandas as pd

from flight_delay.commands.publish import repo_artifacts
from flight_delay.serving.carriers import NAMES, label
from flight_delay.serving.features import PriorRates


def test_every_carrier_in_the_data_has_a_name() -> None:
    """A code with no name shows as a bare 'YX' in the dropdown, which is
    exactly the problem this mapping exists to fix."""
    priors = PriorRates.from_frame(pd.read_parquet(repo_artifacts() / "model" / "priors.parquet"))
    missing = sorted(set(priors.carrier_rate) - set(NAMES))
    assert not missing, f"no name for: {missing}"


def test_the_code_comes_first() -> None:
    """The code is what the feed reports and what the model sees; the name is
    decoration for a human reading a dropdown."""
    assert label("AA") == "AA (American Airlines)"
    assert label("9E") == "9E (Endeavor Air)"
    assert label("OO") == "OO (SkyWest Airlines)"


def test_lowercase_input_is_tolerated() -> None:
    assert label("aa") == "aa (American Airlines)"


def test_an_unknown_code_passes_through_rather_than_raising() -> None:
    """A carrier that starts reporting next year must not break a dropdown."""
    assert label("ZZ") == "ZZ"


def test_no_duplicate_names() -> None:
    assert len(set(NAMES.values())) == len(NAMES)
