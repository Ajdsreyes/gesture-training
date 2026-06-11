from __future__ import annotations

import warnings
import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Sequence flattening
# ---------------------------------------------------------------------------

def _flatten(X_seq: np.ndarray) -> np.ndarray:
    """Flatten a (T, F) observation matrix into a 1-D vector."""
    return X_seq.flatten()


# ---------------------------------------------------------------------------
# EER threshold  (identical logic to hmm_trainer.py)
# ---------------------------------------------------------------------------

def compute_eer_threshold(
    genuine_scores:  np.ndarray,
    impostor_scores: np.ndarray,
    n_thresholds:    int = 1000,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (theta_eer, eer, far_curve, frr_curve). Accept when score >= threshold."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    lo, hi = all_scores.min(), all_scores.max()

    thresholds = np.linspace(lo - 1e-6, hi + 1e-6, n_thresholds)
    far_curve  = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frr_curve  = np.array([np.mean(genuine_scores  <  t) for t in thresholds])

    idx       = int(np.argmin(np.abs(far_curve - frr_curve)))
    theta_eer = float(thresholds[idx])
    eer       = float((far_curve[idx] + frr_curve[idx]) / 2)
    return theta_eer, eer, far_curve, frr_curve


# ---------------------------------------------------------------------------
# Nu hyper-parameter selection via cross-validation
# (mirrors select_n_states_cv in hmm_trainer.py)
# ---------------------------------------------------------------------------

def select_nu_cv(
    X_flat:        np.ndarray,
    candidate_nus: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
    n_splits:      int = 5,
    kernel:        str = "rbf",
) -> Tuple[float, Dict[float, float]]:
    """
    Choose the best nu for OneClassSVM via K-Fold CV on genuine training data.
    Scoring: mean decision-function value on held-out genuine sequences.
    Returns (best_nu, {nu: mean_cv_score}).
    """
    n_seqs = X_flat.shape[0]
    kf     = KFold(n_splits=min(n_splits, n_seqs), shuffle=True, random_state=42)

    cv_scores: Dict[float, List[float]] = {nu: [] for nu in candidate_nus}

    for train_idx, val_idx in kf.split(np.arange(n_seqs)):
        X_tr  = X_flat[train_idx]
        X_val = X_flat[val_idx]
        if len(X_tr) == 0 or len(X_val) == 0:
            continue
        for nu in candidate_nus:
            try:
                clf    = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
                clf.fit(X_tr)
                scores = clf.decision_function(X_val)
                finite = scores[np.isfinite(scores)]
                cv_scores[nu].append(float(np.mean(finite)) if len(finite) > 0 else -np.inf)
            except Exception:
                cv_scores[nu].append(-np.inf)

    mean_scores: Dict[float, float] = {}
    for nu, vals in cv_scores.items():
        finite = [v for v in vals if np.isfinite(v)]
        mean_scores[nu] = float(np.mean(finite)) if finite else -np.inf

    best_nu = max(mean_scores, key=lambda k: mean_scores[k])
    return best_nu, mean_scores


# ---------------------------------------------------------------------------
# Core model wrapper  (mirrors LeftRightHMM)
# ---------------------------------------------------------------------------

class OneClassSVMModel:
    """
    Thin wrapper around sklearn OneClassSVM.
    Mirrors the LeftRightHMM interface: fit / decision_score / score_sequences.
    """

    def __init__(self, nu: float = 0.1, kernel: str = "rbf"):
        self.nu        = nu
        self.kernel    = kernel
        self._clf: Optional[OneClassSVM] = None
        self.is_fitted = False

    def fit(self, X_flat: np.ndarray) -> "OneClassSVMModel":
        """X_flat: (n_seqs, n_features) — one flattened row per GestureSequence."""
        self._clf = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma="scale")
        self._clf.fit(X_flat)
        self.is_fitted = True
        return self

    def decision_score(self, x_flat: np.ndarray) -> float:
        """
        Signed distance from the SVM hyperplane for a single flattened vector.
        Positive = genuine-like (accept), negative = anomaly (reject).
        Semantically equivalent to HMM log-likelihood: higher == more genuine.
        """
        if not self.is_fitted or self._clf is None:
            raise RuntimeError("Call fit() before decision_score().")
        try:
            val = float(self._clf.decision_function(x_flat.reshape(1, -1))[0])
            return val if np.isfinite(val) else -1e6
        except Exception:
            return -1e6

    def score_sequences(self, X_flat: np.ndarray) -> np.ndarray:
        """Score a batch of flattened sequences; returns (n,) array."""
        return np.array([self.decision_score(row) for row in X_flat])


# ---------------------------------------------------------------------------
# Per-user model  (mirrors UserHMM)
# ---------------------------------------------------------------------------

class UserSVM:
    """
    Per-user One-Class SVM authenticator.
    Drop-in replacement for UserHMM — identical public API:
    fit / score / set_threshold / authenticate.
    """

    def __init__(self, participant_id: str, nu: float = 0.1, kernel: str = "rbf"):
        self.participant_id = participant_id
        self.svm            = OneClassSVMModel(nu=nu, kernel=kernel)
        self.threshold: Optional[float] = None

    def fit(self, X_flat: np.ndarray) -> "UserSVM":
        """X_flat: (n_seqs, n_features) — one row per GestureSequence."""
        self.svm.fit(X_flat)
        return self

    def score(self, X_seq: np.ndarray) -> float:
        """Score a single (T, F) observation matrix."""
        return self.svm.decision_score(_flatten(X_seq))

    def set_threshold(self, theta: float):
        self.threshold = theta

    def authenticate(self, X_seq: np.ndarray) -> Tuple[bool, float]:
        if self.threshold is None:
            raise RuntimeError("set_threshold() must be called before authenticate().")
        s = self.score(X_seq)
        return (s >= self.threshold), s


# ---------------------------------------------------------------------------
# Helper: build flat feature matrix from a list of GestureSequences
# ---------------------------------------------------------------------------

def sequences_to_flat(sequences: list, extractor, normaliser) -> np.ndarray:
    """
    Convert a list of GestureSequence objects -> (n_seqs, T*F) array.
    Each row is the flattened, normalised observation matrix of one sequence.
    """
    rows = []
    for s in sequences:
        mat = extractor.sequence_to_matrix(s)
        mat = normaliser.transform(mat)
        rows.append(_flatten(mat))
    return np.vstack(rows).astype(np.float64)


# ---------------------------------------------------------------------------
# Main training entry-point  (mirrors train_user_model in hmm_trainer.py)
# ---------------------------------------------------------------------------

def train_user_model(
    participant_id:    str,
    train_sequences:   list,
    val_genuine_seqs:  list,
    val_impostor_seqs: list,
    extractor,
    normaliser,
    candidate_nus:     List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
    kernel:            str = "rbf",
    verbose:           bool = False,
) -> UserSVM:
    """
    Train a per-user One-Class SVM, select nu via cross-validation, and
    set an EER threshold from validation genuine/impostor scores.

    Parameters mirror train_user_model() in hmm_trainer.py exactly,
    with candidate_nus/kernel replacing candidate_states/n_iter.
    """
    X_train = sequences_to_flat(train_sequences, extractor, normaliser)

    best_nu, cv_scores = select_nu_cv(X_train, candidate_nus=candidate_nus, kernel=kernel)
    if verbose:
        print(f"  [{participant_id}] CV scores: {cv_scores}  -> best nu={best_nu}")

    user_model = UserSVM(participant_id=participant_id, nu=best_nu, kernel=kernel)
    user_model.fit(X_train)

    def _score_list(seqs: list) -> np.ndarray:
        scores = []
        for s in seqs:
            mat = extractor.sequence_to_matrix(s)
            mat = normaliser.transform(mat)
            scores.append(user_model.score(mat))
        return np.array(scores)

    gen_scores = _score_list(val_genuine_seqs)
    imp_scores = _score_list(val_impostor_seqs)

    gen_finite = gen_scores[np.isfinite(gen_scores)]
    imp_finite = imp_scores[np.isfinite(imp_scores)]

    if len(gen_finite) == 0 or len(imp_finite) == 0:
        train_scores = user_model.svm.score_sequences(X_train)
        finite_train = train_scores[np.isfinite(train_scores)]
        theta = float(np.percentile(finite_train, 10)) if len(finite_train) > 0 else -1e3
        if verbose:
            print(f"  [{participant_id}] WARNING: degenerate val scores, "
                  f"using training-based theta={theta:.3f}")
    else:
        theta, eer, _, _ = compute_eer_threshold(gen_finite, imp_finite)
        if verbose:
            print(f"  [{participant_id}] EER={eer:.4f}  threshold={theta:.4f}")

    user_model.set_threshold(theta)
    return user_model


# ---------------------------------------------------------------------------
# Smoke-test  (mirrors __main__ in hmm_trainer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import defaultdict
    from gesture_data      import generate_synthetic_dataset
    from feature_extractor import FeatureExtractor, ZScoreNormaliser

    print("Generating synthetic dataset ...")
    all_seqs = generate_synthetic_dataset(n_participants=10, n_sessions=3, n_repetitions=10)

    by_pid: Dict[str, list] = defaultdict(list)
    for s in all_seqs:
        by_pid[s.participant_id].append(s)

    extractor = FeatureExtractor()
    pids      = list(by_pid.keys())
    enrolled  = pids[0]

    train_seqs = [s for s in by_pid[enrolled] if s.session_id == 1]
    val_gen    = [s for s in by_pid[enrolled] if s.session_id == 2]
    val_imp    = [s for p in pids[1:5] for s in by_pid[p] if s.session_id == 2]

    X_all, _ = extractor.sequences_to_arrays(train_seqs)
    norm = ZScoreNormaliser()
    norm.fit(X_all)

    print(f"Training SVM user model for {enrolled} ...")
    model = train_user_model(
        participant_id    = enrolled,
        train_sequences   = train_seqs,
        val_genuine_seqs  = val_gen,
        val_impostor_seqs = val_imp,
        extractor         = extractor,
        normaliser        = norm,
        verbose           = True,
    )

    test_seqs = [s for s in by_pid[enrolled] if s.session_id == 3]
    print()
    for seq in test_seqs[:5]:
        mat          = norm.transform(extractor.sequence_to_matrix(seq))
        accepted, sc = model.authenticate(mat)
        print(f"  Genuine seq: accepted={accepted}  score={sc:.4f}  threshold={model.threshold:.4f}")

    print("\nSVM trainer OK.")