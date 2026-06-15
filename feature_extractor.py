from __future__ import annotations
import numpy as np
from typing import List, Optional

from gesture_data import (
    Gesture,
    GestureSequence,
    MULTI_TOUCH_GESTURES,
    TouchEvent,
)


# Feature names 

FEATURE_NAMES = [
    "mid_stroke_area_covered",           # 20.58%
    "pairwise_velocity_20pct",           # 19.63%
    "mid_stroke_pressure",               # 17.28%
    "direction_end_to_end",              # 11.06%
    "stop_x",                            # 10.32%
    "start_x",                           # 10.15%
    "average_direction",                 #  9.45%
    "start_y",                           #  9.43%
    "average_velocity",                  #  8.84%
    "stop_y",                            #  8.61%
    "stroke_duration",                   #  8.50%
    "direct_end_to_end_distance",        #  8.27%
    "length_of_trajectory",              #  8.16%
    "pairwise_velocity_80pct",           #  7.85%
    "median_velocity_last3pts",          #  7.24%
    "pairwise_velocity_50pct",           #  7.22%
    "pairwise_acc_20pct",                #  7.07%
    "ratio_end_to_end_vs_trajectory",    #  6.29%
    "largest_deviation_end_to_end",      #  6.08%
    "pairwise_acc_80pct",                #  5.96%
    "mean_resultant_length",             #  5.82%
    "median_acc_first5pts",              #  5.42%
    "deviation_end_to_end_50pct",        #  5.39%
    "inter_stroke_time",                 #  5.30%
    "deviation_end_to_end_80pct",        #  5.14%
    "deviation_end_to_end_20pct",        #  5.04%
    "pairwise_acc_50pct",                #  3.44%
    "phone_orientation",                 #  3.08%
    "mid_stroke_finger_orientation",     #  0.97%
    "up_down_left_right_flag",           #  0.00%
    "change_of_finger_orientation",      #  0.00%
]

N_FEATURES = len(FEATURE_NAMES)  # 31


# Low-level helpers

def _pairwise_velocities(events: List[TouchEvent]) -> np.ndarray:
    """Speed between consecutive touch events."""
    if len(events) < 2:
        return np.array([0.0])
    speeds = []
    for i in range(1, len(events)):
        dt = events[i].timestamp - events[i - 1].timestamp
        dx = events[i].x - events[i - 1].x
        dy = events[i].y - events[i - 1].y
        speed = float(np.hypot(dx, dy) / dt) if dt > 1e-9 else 0.0
        speeds.append(speed)
    return np.array(speeds)

def _pairwise_accelerations(events: List[TouchEvent]) -> np.ndarray:
    """Acceleration between consecutive velocity samples."""
    vels = _pairwise_velocities(events)
    if len(vels) < 2:
        return np.array([0.0])
    accs = []
    for i in range(1, len(vels)):
        dt = (
            events[i + 1].timestamp - events[i].timestamp
            if (i + 1) < len(events)
            else 1.0
        )
        accs.append((vels[i] - vels[i - 1]) / dt if dt > 1e-9 else 0.0)
    return np.array(accs)

def _trajectory_length(events: List[TouchEvent]) -> float:
    """Total arc length of the touch path."""
    total = 0.0
    for i in range(1, len(events)):
        total += float(np.hypot(
            events[i].x - events[i - 1].x,
            events[i].y - events[i - 1].y,
        ))
    return total

def _deviations_from_end_to_end(events: List[TouchEvent]) -> np.ndarray:
    """
    Perpendicular distance of each point from the straight line
    connecting the first and last touch event.
    """
    if len(events) < 2:
        return np.array([0.0])
    sx, sy = events[0].x, events[0].y
    ex, ey = events[-1].x, events[-1].y
    disp   = float(np.hypot(ex - sx, ey - sy))
    if disp < 1e-6:
        return np.zeros(len(events))
    devs = []
    for e in events:
        cross = abs((ey - sy) * e.x - (ex - sx) * e.y + ex * sy - ey * sx)
        devs.append(cross / disp)
    return np.array(devs)

def _direction_of_end_to_end(events: List[TouchEvent]) -> float:
    """Angle (radians) of the vector from first to last point."""
    if len(events) < 2:
        return 0.0
    return float(np.arctan2(
        events[-1].y - events[0].y,
        events[-1].x - events[0].x,
    ))

def _point_directions(events: List[TouchEvent]) -> np.ndarray:
    """Bearing angle of each consecutive segment."""
    if len(events) < 2:
        return np.array([0.0])
    dirs = []
    for i in range(1, len(events)):
        dx = events[i].x - events[i - 1].x
        dy = events[i].y - events[i - 1].y
        dirs.append(float(np.arctan2(dy, dx)))
    return np.array(dirs)

def _mean_resultant_length(angles: np.ndarray) -> float:
    """
    Circular mean resultant length.
    1.0 = perfectly straight motion, 0.0 = random direction changes.
    """
    if len(angles) == 0:
        return 0.0
    return float(np.sqrt(
        np.mean(np.cos(angles)) ** 2 +
        np.mean(np.sin(angles)) ** 2
    ))

def _mid_index(events: List[TouchEvent]) -> int:
    return max(0, len(events) // 2)


def _safe_percentile(arr: np.ndarray, pct: float) -> float:
    """Percentile with fallback to 0 for empty arrays."""
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, pct))


# Main feature extractor

class FeatureExtractor:

    def gesture_to_vector(self, gesture: Gesture) -> np.ndarray:
        events = gesture.events

        if len(events) < 2:
            return np.zeros(N_FEATURES, dtype=np.float64)

        # ── Basic positions ─────────────────────────────────────────────────
        sx, sy  = events[0].x,  events[0].y
        ex, ey  = events[-1].x, events[-1].y
        mid_idx = _mid_index(events)
        mid_evt = events[mid_idx]

        # ── Timing ──────────────────────────────────────────────────────────
        duration         = events[-1].timestamp - events[0].timestamp
        inter_stroke_time = float(events[1].timestamp - events[0].timestamp) \
            if len(events) >= 2 else 0.0

        # ── Spatial ─────────────────────────────────────────────────────────
        traj_len  = _trajectory_length(events)
        e2e_dist  = float(np.hypot(ex - sx, ey - sy))
        ratio_e2e = (e2e_dist / traj_len) if traj_len > 1e-6 else 1.0

        # ── Velocity ────────────────────────────────────────────────────────
        pw_vels      = _pairwise_velocities(events)
        avg_velocity = float(pw_vels.mean()) if len(pw_vels) > 0 else 0.0

        last3_vels       = pw_vels[-3:] if len(pw_vels) >= 3 else pw_vels
        median_vel_last3 = float(np.median(last3_vels)) if len(last3_vels) > 0 else 0.0

        # ── Acceleration ─────────────────────────────────────────────────────
        pw_accs = _pairwise_accelerations(events)

        first5_accs       = pw_accs[:5] if len(pw_accs) >= 5 else pw_accs
        median_acc_first5 = float(np.median(first5_accs)) if len(first5_accs) > 0 else 0.0

        # ── Direction ────────────────────────────────────────────────────────
        dirs          = _point_directions(events)
        avg_direction = float(np.mean(dirs)) if len(dirs) > 0 else 0.0

        # ── Deviation from end-to-end line ───────────────────────────────────
        deviations  = _deviations_from_end_to_end(events)
        largest_dev = float(deviations.max()) if len(deviations) > 0 else 0.0

        # ── Pressure ─────────────────────────────────────────────────────────
        pressures        = np.array([e.pressure for e in events])
        mid_pressure     = float(pressures[mid_idx])

        # ── Mid-stroke area covered ──────────────────────────────────────────
        # Bounding box area of the middle third of the stroke
        n      = len(events)
        lo_idx = max(0, n // 3)
        hi_idx = min(n, 2 * n // 3)
        mid_events = events[lo_idx:hi_idx] if hi_idx > lo_idx else events
        mid_xs     = np.array([e.x for e in mid_events])
        mid_ys     = np.array([e.y for e in mid_events])
        mid_area   = float(
            (mid_xs.max() - mid_xs.min()) * (mid_ys.max() - mid_ys.min())
        ) if len(mid_events) > 1 else 0.0

        # ── Mean resultant length ─────────────────────────────────────────────
        mrl = _mean_resultant_length(dirs)

        # ── Phone orientation ─────────────────────────────────────────────────
        # Derived from bounding box aspect ratio of the full stroke
        all_xs = np.array([e.x for e in events])
        all_ys = np.array([e.y for e in events])
        bb_w   = float(all_xs.max() - all_xs.min())
        bb_h   = float(all_ys.max() - all_ys.min())
        phone_orientation = float(np.arctan2(bb_h, bb_w)) \
            if (bb_w + bb_h) > 1e-6 else 0.0

        # ── Mid-stroke finger orientation ─────────────────────────────────────
        # Angle of motion at the midpoint
        if mid_idx > 0:
            mid_finger_orient = float(np.arctan2(
                mid_evt.y - events[mid_idx - 1].y,
                mid_evt.x - events[mid_idx - 1].x,
            ))
        else:
            mid_finger_orient = 0.0

        # ── Change of finger orientation ──────────────────────────────────────
        # Standard deviation of segment directions
        change_finger_orient = float(np.std(dirs)) if len(dirs) > 1 else 0.0

        # ── Up/down/left/right flag ───────────────────────────────────────────
        #  1 = right,  -1 = left,  2 = down,  -2 = up
        dx_total = ex - sx
        dy_total = ey - sy
        if abs(dx_total) >= abs(dy_total):
            ud_lr_flag = 1.0 if dx_total >= 0 else -1.0
        else:
            ud_lr_flag = 2.0 if dy_total >= 0 else -2.0

        # ── Assemble in exact paper order ─────────────────────────────────────
        vec = np.array([
            mid_area,                               # mid_stroke_area_covered
            _safe_percentile(pw_vels, 20),          # pairwise_velocity_20pct
            mid_pressure,                           # mid_stroke_pressure
            _direction_of_end_to_end(events),       # direction_end_to_end
            ex,                                     # stop_x
            sx,                                     # start_x
            avg_direction,                          # average_direction
            sy,                                     # start_y
            avg_velocity,                           # average_velocity
            ey,                                     # stop_y
            duration,                               # stroke_duration
            e2e_dist,                               # direct_end_to_end_distance
            traj_len,                               # length_of_trajectory
            _safe_percentile(pw_vels, 80),          # pairwise_velocity_80pct
            median_vel_last3,                       # median_velocity_last3pts
            _safe_percentile(pw_vels, 50),          # pairwise_velocity_50pct
            _safe_percentile(pw_accs, 20),          # pairwise_acc_20pct
            ratio_e2e,                              # ratio_end_to_end_vs_trajectory
            largest_dev,                            # largest_deviation_end_to_end
            _safe_percentile(pw_accs, 80),          # pairwise_acc_80pct
            mrl,                                    # mean_resultant_length
            median_acc_first5,                      # median_acc_first5pts
            _safe_percentile(deviations, 50),       # deviation_end_to_end_50pct
            inter_stroke_time,                      # inter_stroke_time
            _safe_percentile(deviations, 80),       # deviation_end_to_end_80pct
            _safe_percentile(deviations, 20),       # deviation_end_to_end_20pct
            _safe_percentile(pw_accs, 50),          # pairwise_acc_50pct
            phone_orientation,                      # phone_orientation
            mid_finger_orient,                      # mid_stroke_finger_orientation
            ud_lr_flag,                             # up_down_left_right_flag
            change_finger_orient,                   # change_of_finger_orientation
        ], dtype=np.float64)

        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def sequence_to_matrix(self, seq: GestureSequence) -> np.ndarray:
        """Returns shape (n_gestures, 31)."""
        rows = [self.gesture_to_vector(g) for g in seq.gestures]
        return np.vstack(rows).astype(np.float64)

    def sequences_to_arrays(self, sequences: List[GestureSequence]):
        """
        Converts a list of GestureSequence objects to a stacked
        feature matrix X of shape (n_sequences * n_gestures, 31).
        """
        matrices = [self.sequence_to_matrix(s) for s in sequences]
        X        = np.vstack(matrices).astype(np.float64)
        lengths  = [m.shape[0] for m in matrices]
        return X, lengths


# Z-score normaliser

class ZScoreNormaliser:

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreNormaliser":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
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


# Smoke test

if __name__ == "__main__":
    from gesture_data import generate_synthetic_dataset

    print(f"Feature count : {N_FEATURES}")
    print()

    print("Generating synthetic dataset (5 participants) ...")
    all_seqs = generate_synthetic_dataset(n_participants=5)

    extractor = FeatureExtractor()

    seq = all_seqs[0]
    mat = extractor.sequence_to_matrix(seq)
    print(f"Sequence      : {seq}")
    print(f"Matrix shape  : {mat.shape}  (n_gestures × {N_FEATURES} features)")
    print()
    print("Feature values for gesture 0:")
    for name, val in zip(FEATURE_NAMES, mat[0]):
        print(f"  {name:<42} {val:>12.4f}")

    X, lengths = extractor.sequences_to_arrays(all_seqs[:10])
    print(f"\nBatch X shape : {X.shape}")
    print(f"Unique lengths: {set(lengths)}")

    norm   = ZScoreNormaliser()
    X_norm = norm.fit_transform(X)
    print(f"\nNormalised mean (should be ~0): {X_norm.mean(axis=0).round(2)}")
    print(f"Normalised std  (should be ~1): {X_norm.std(axis=0).round(2)}")
    print("\nFeature extraction OK.")