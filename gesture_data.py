from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class TouchEvent:

    timestamp : float
    x         : float
    y         : float
    pressure  : float
    finger_id : int = 0

SINGLE_TOUCH_GESTURES = {"tap", "swipe", "scroll"}
MULTI_TOUCH_GESTURES  = {"zoom", "pinch"}
VALID_GESTURE_TYPES   = SINGLE_TOUCH_GESTURES | MULTI_TOUCH_GESTURES
VALID_ORIENTATIONS    = {"horizontal", "vertical"}


@dataclass
class Gesture:

    gesture_type   : str
    orientation    : str
    events         : List[TouchEvent] = field(default_factory=list)
    session_id     : int = 1
    participant_id : str = "unknown"
    repetition     : int = 0

    def __post_init__(self):
        if self.gesture_type not in VALID_GESTURE_TYPES:
            raise ValueError(
                f"gesture_type must be one of {VALID_GESTURE_TYPES}, "
                f"got '{self.gesture_type}'"
            )
        if self.orientation not in VALID_ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {VALID_ORIENTATIONS}, "
                f"got '{self.orientation}'"
            )
        
        self.events.sort(key=lambda e: e.timestamp)

    @property
    def is_multitouch(self) -> bool:
        return self.gesture_type in MULTI_TOUCH_GESTURES

    @property
    def duration_ms(self) -> float:
        if len(self.events) < 2:
            return 0.0
        return self.events[-1].timestamp - self.events[0].timestamp

@dataclass
class GestureSequence:
    
    gestures       : List[Gesture]
    participant_id : str = "unknown"
    session_id     : int = 1
    sequence_label : str = ""

    def __post_init__(self):
        if len(self.gestures) != 3:
            raise ValueError(
                f"GestureSequence must contain exactly 3 gestures, "
                f"got {len(self.gestures)}"
            )
        if not self.sequence_label:
            self.sequence_label = "-".join(g.gesture_type for g in self.gestures)

    def __repr__(self) -> str:
        return (
            f"GestureSequence(label='{self.sequence_label}', "
            f"participant='{self.participant_id}', session={self.session_id})"
        )

import numpy as np

def _make_swipe_events(
    n_points: int = 20,
    duration_ms: float = 400,
    start: tuple = (100, 300),
    end:   tuple = (600, 300),
    pressure_mean: float = 0.5,
    noise: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> List[TouchEvent]:
    
    if rng is None:
        rng = np.random.default_rng()

    timestamps = np.linspace(0, duration_ms, n_points)
    xs = np.linspace(start[0], end[0], n_points) + rng.normal(0, noise, n_points)
    ys = np.linspace(start[1], end[1], n_points) + rng.normal(0, noise, n_points)
    pressures = rng.normal(pressure_mean, 0.05, n_points).clip(0.1, 1.0)

    return [
        TouchEvent(float(t), float(x), float(y), float(p))
        for t, x, y, p in zip(timestamps, xs, ys, pressures)
    ]

def _make_tap_events(
    x: float = 300,
    y: float = 400,
    duration_ms: float = 80,
    pressure_mean: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> List[TouchEvent]:

    if rng is None:
        rng = np.random.default_rng()
    n = rng.integers(2, 4)
    return [
        TouchEvent(
            t,
            x + rng.normal(0, 2),
            y + rng.normal(0, 2),
            float(np.clip(rng.normal(pressure_mean, 0.05), 0.1, 1.0)),
        )
        for t in np.linspace(0, duration_ms, n)
    ]


def _make_pinch_events(
    centre: tuple = (400, 400),
    start_spread: float = 200,
    end_spread: float = 80,
    duration_ms: float = 500,
    n_points: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> List[TouchEvent]:

    if rng is None:
        rng = np.random.default_rng()

    spreads = np.linspace(start_spread, end_spread, n_points)
    timestamps = np.linspace(0, duration_ms, n_points)
    events = []
    for t, s in zip(timestamps, spreads):
        events.append(TouchEvent(float(t), centre[0] - s/2, centre[1],
                                 float(np.clip(rng.normal(0.5, 0.05), 0.1, 1.0)),
                                 finger_id=0))
        events.append(TouchEvent(float(t), centre[0] + s/2, centre[1],
                                 float(np.clip(rng.normal(0.5, 0.05), 0.1, 1.0)),
                                 finger_id=1))
    return events


def generate_synthetic_dataset(
    n_participants: int = 20,
    n_sessions: int = 3,
    n_repetitions: int = 10,
    seed: int = 42,
) -> List[GestureSequence]:
    
    rng = np.random.default_rng(seed)
    sequences: List[GestureSequence] = []

    # One fixed 3-gesture combination: tap -> swipe -> pinch
    SEQUENCE_TYPES = [
        ("tap", "horizontal"),
        ("swipe", "horizontal"),
        ("pinch", "horizontal"),
    ]

    for p_idx in range(n_participants):
        pid = f"P{p_idx:03d}"
        # Per-participant behavioural profile
        speed_factor   = rng.uniform(0.7, 1.4)
        pressure_bias  = rng.uniform(-0.1, 0.1)
        spread_factor  = rng.uniform(0.8, 1.3)

        for session in range(1, n_sessions + 1):
            # Small session-level drift
            session_noise = rng.normal(0, 0.02)

            for rep in range(1, n_repetitions + 1):
                gestures = []

                for g_type, orientation in SEQUENCE_TYPES:
                    if g_type == "tap":
                        evs = _make_tap_events(
                            x=300 + rng.normal(0, 5),
                            y=400 + rng.normal(0, 5),
                            duration_ms=80 / speed_factor,
                            pressure_mean=0.6 + pressure_bias + session_noise,
                            rng=rng,
                        )
                    elif g_type == "swipe":
                        evs = _make_swipe_events(
                            duration_ms=400 / speed_factor,
                            start=(100, 300),
                            end=(600, 300),
                            pressure_mean=0.5 + pressure_bias + session_noise,
                            rng=rng,
                        )
                    else:  # pinch
                        evs = _make_pinch_events(
                            start_spread=200 * spread_factor,
                            end_spread=80 * spread_factor,
                            duration_ms=500 / speed_factor,
                            rng=rng,
                        )

                    gestures.append(Gesture(
                        gesture_type=g_type,
                        orientation=orientation,
                        events=evs,
                        session_id=session,
                        participant_id=pid,
                        repetition=rep,
                    ))

                sequences.append(GestureSequence(
                    gestures=gestures,
                    participant_id=pid,
                    session_id=session,
                ))

    return sequences