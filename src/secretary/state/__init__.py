"""Continuing agent core state layer.

``Reducer`` owns the deterministic working-state transition. Everything
else lives under its parent packages; this package exists to make the
core transition testable in isolation from perception and policy.
"""

from .reducer import Observation, WorldStateReducer, to_observation

__all__ = ["Observation", "WorldStateReducer", "to_observation"]
