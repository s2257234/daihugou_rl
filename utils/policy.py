"""Policy/temperature utilities.

SRP: keep small action-selection math helpers out of agent implementations.
"""

from __future__ import annotations

from typing import List


def softmax_temperature_policy(visits: List[int], temperature: float) -> List[float]:
    """Apply temperature to visit counts and return a probability distribution.

    When temperature is near 0, returns an (almost) deterministic one-hot at argmax.
    """
    if not visits:
        return []
    if temperature <= 1e-6:
        m = max(visits)
        return [1.0 if v == m else 0.0 for v in visits]
    scaled = [v ** (1.0 / max(temperature, 1e-6)) for v in visits]
    s = sum(scaled)
    return [x / s for x in scaled] if s > 0 else [1.0 / len(scaled)] * len(scaled)
