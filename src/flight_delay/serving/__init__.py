"""Serving the pre-departure model.

Only scenario A is served. Scenario B scores far better and answers a question
nobody needs answered at request time: by then the aircraft has already left
and its departure delay is known.
"""
