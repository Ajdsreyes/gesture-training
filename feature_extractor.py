from __future__ import annotations
import numpy as np
from typing import List, Optional

from gesture_data import (
    Gesture, GestureSequence,
    MULTI_TOUCH_GESTURES,
    TouchEvent,
)

N_FEATURES = 15

FEATURE_NAMES = [
    "duration",
    "start_x", "start_y", "end_x", "end_y",
    "mean_pressure", "pressure_std", "displacement",
    "mean_speed", "speed_std",
    "slope_e2e", "mean_dev_e2e",
    "start_inter_finger", "end_inter_finger", "scale_factor",
]

def _split_fingers(events: List[TouchEvent]):
    f0 = sorted([e for e in events if e.finger_id == 0], key=lambda e: e.timestamp)
    f1 = sorted([e for e in events if e.finger_id == 1], key=lambda e: e.timestamp)
    return f0, f1


def _preprocess_events(events: List[TouchEvent]) -> List[TouchEvent]:
    """Return a new event list with:
    - timestamps rebased to 0 at the first event
    - x/y coordinates normalised to [0, 1] within the gesture's bounding box
      (degenerate axes where min == max are left at 0.0)
    """
    if not events:
        return events

    t0   = events[0].timestamp
    xs   = np.array([e.x for e in events])
    ys   = np.array([e.y for e in events])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min if x_max - x_min > 1e-6 else 1.0
    y_range = y_max - y_min if y_max - y_min > 1e-6 else 1.0

    return [
        TouchEvent(
            timestamp = e.timestamp - t0,
            x         = (e.x - x_min) / x_range,
            y         = (e.y - y_min) / y_range,
            pressure  = e.pressure,
            finger_id = e.finger_id,
        )
        for e in events
    ]


def _core_features(events: List[TouchEvent]) -> np.ndarray:
    if len(events) < 2:
        return np.zeros(8)

    duration = events[-1].timestamp - events[0].timestamp
    sx, sy   = events[0].x, events[0].y
    ex, ey   = events[-1].x, events[-1].y

    pres        = np.array([e.pressure for e in events])
    mean_pres   = float(pres.mean())
    std_pres    = float(pres.std())
    displacement = float(np.hypot(ex - sx, ey - sy))

    return np.array([duration, sx, sy, ex, ey, mean_pres, std_pres, displacement])


def _dynamic_features(events: List[TouchEvent]) -> np.ndarray:
    if len(events) < 2:
        return np.zeros(4)

    sx, sy = events[0].x, events[0].y
    ex, ey = events[-1].x, events[-1].y
    disp   = float(np.hypot(ex - sx, ey - sy))

    speeds = []
    for i in range(1, len(events)):
        dt = events[i].timestamp - events[i - 1].timestamp
        if dt > 0:
            dx = events[i].x - events[i - 1].x
            dy = events[i].y - events[i - 1].y
            speeds.append(float(np.hypot(dx, dy) / dt))

    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    speed_std  = float(np.std(speeds))  if speeds else 0.0

    slope_e2e = float(np.arctan2(ey - sy, ex - sx))

    if disp > 1e-6:
        devs = []
        for e in events:
            cross = abs((ey - sy) * e.x - (ex - sx) * e.y + ex * sy - ey * sx)
            devs.append(cross / disp)
        mean_dev = float(np.mean(devs))
    else:
        mean_dev = 0.0

    return np.array([mean_speed, speed_std, slope_e2e, mean_dev])


def _multitouch_features(
    f0: List[TouchEvent], f1: List[TouchEvent]
) -> np.ndarray:

    if not f0 or not f1:
        return np.array([0.0, 0.0, 1.0])

    start_dist = float(np.hypot(f0[0].x - f1[0].x, f0[0].y - f1[0].y))
    end_dist   = float(np.hypot(f0[-1].x - f1[-1].x, f0[-1].y - f1[-1].y))
    scale      = (end_dist / start_dist) if start_dist > 1e-6 else 1.0

    return np.array([start_dist, end_dist, scale])


class FeatureExtractor:

    def gesture_to_vector(self, gesture: Gesture) -> np.ndarray:

        events = _preprocess_events(gesture.events)

        if gesture.is_multitouch:
            # Split after joint preprocessing so both fingers share the same
            # bounding box and time origin, preserving inter-finger geometry.
            f0, f1 = _split_fingers(events)
            primary = f0 if f0 else events  # fall back gracefully
            core    = _core_features(primary)
            dynamic = _dynamic_features(primary)
            multi   = _multitouch_features(f0, f1)
        else:
            core    = _core_features(events)
            dynamic = _dynamic_features(events)
            multi   = np.array([0.0, 0.0, 1.0])  # neutral placeholders

        vec = np.concatenate([core, dynamic, multi])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec.astype(np.float64)

    def sequence_to_matrix(self, seq: GestureSequence) -> np.ndarray:

        rows = [self.gesture_to_vector(g) for g in seq.gestures]
        return np.vstack(rows).astype(np.float64)   # shape (3, 15)

    def sequences_to_arrays(self, sequences: List[GestureSequence]):

        matrices = [self.sequence_to_matrix(s) for s in sequences]
        X       = np.vstack(matrices).astype(np.float64)
        lengths = [m.shape[0] for m in matrices]  # each is 3
        return X, lengths

class ZScoreNormaliser:

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreNormaliser":

        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        # Replace zero std with 1 to avoid division by zero
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

if __name__ == "__main__":
    from gesture_data import generate_synthetic_dataset

    print("Generating 20-participant synthetic dataset ...")
    all_seqs = generate_synthetic_dataset(n_participants=20)

    extractor = FeatureExtractor()

    seq = all_seqs[0]
    mat = extractor.sequence_to_matrix(seq)
    print(f"Sequence: {seq}")
    print(f"Observation matrix shape: {mat.shape}")
    print(f"Feature names: {FEATURE_NAMES}")
    print(f"First gesture features:\n{mat[0].round(3)}")

    X, lengths = extractor.sequences_to_arrays(all_seqs[:30])
    print(f"\nBatch X shape: {X.shape}  (30 sequences x 3 steps)")
    print(f"Unique lengths: {set(lengths)}")

    norm = ZScoreNormaliser()
    X_norm = norm.fit_transform(X)
    print(f"\nNormalised X  mean: {X_norm.mean(axis=0).round(3)}")
    print(f"Normalised X  std : {X_norm.std(axis=0).round(3)}")
    print("\nFeature extraction OK.")