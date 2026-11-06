"""Bearing fault frequencies from geometry.

A defect on a rolling element bearing is struck at a rate set by the bearing's shape and
the shaft speed, so each fault type has its own "signature" frequency:

    BPFO  ball pass frequency, outer race   (a ball rolls over an outer race defect)
    BPFI  ball pass frequency, inner race   (a ball rolls over an inner race defect)
    BSF   ball spin frequency               (a ball defect turning on its own axis)
    FTF   fundamental train frequency       (the cage rotation speed)

Convention note: the CWRU web page lists 4.7135 x shaft speed for ball faults. That is
2 x BSF, because a ball defect hits both races on each spin. This module uses the
textbook BSF (2.357 x), and 2 x BSF is covered as its second harmonic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    name: str
    n_balls: int
    ball_diameter: float   # any unit, as long as both diameters use the same one
    pitch_diameter: float
    contact_angle_deg: float = 0.0


# The drive-end bearing on the CWRU test rig (dimensions in inches).
SKF_6205 = BearingGeometry("SKF 6205-2RS JEM", 9, 0.3126, 1.537, 0.0)


def fault_orders(g: BearingGeometry) -> dict[str, float]:
    """Fault frequencies as multiples of shaft speed."""
    ratio = (g.ball_diameter / g.pitch_diameter) * math.cos(math.radians(g.contact_angle_deg))
    return {
        "ftf": 0.5 * (1 - ratio),
        "bpfo": 0.5 * g.n_balls * (1 - ratio),
        "bpfi": 0.5 * g.n_balls * (1 + ratio),
        "bsf": (g.pitch_diameter / (2 * g.ball_diameter)) * (1 - ratio ** 2),
    }


def fault_frequencies(g: BearingGeometry, rpm: float) -> dict[str, float]:
    """Fault frequencies in Hz at a given shaft speed."""
    if rpm <= 0:
        raise ValueError("rpm must be positive")
    shaft_hz = rpm / 60.0
    return {"shaft": shaft_hz, **{k: v * shaft_hz for k, v in fault_orders(g).items()}}
