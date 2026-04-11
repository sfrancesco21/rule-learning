from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np


def _prep(data: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    tr = data["training"]
    te = data["test"]

    x_tr = np.asarray(tr["stimuli"], dtype=float)
    y_tr = np.asarray(tr["labels"], dtype=int)
    a_tr = np.asarray(tr["action"], dtype=float)
    lost_tr = np.asarray(tr["trialLost"], dtype=int)

    x_te = np.asarray(te["stimuli"], dtype=float)
    y_te = np.asarray(te["labels"], dtype=int)
    a_te = np.asarray(te["action"], dtype=float)
    lost_te = np.asarray(te["trialLost"], dtype=int)

    if x_tr.ndim != 2 or x_tr.shape[1] != 2:
        raise ValueError("training stimuli must have shape (N, 2)")
    if x_te.ndim != 2 or x_te.shape[1] != 2:
        raise ValueError("test stimuli must have shape (N, 2)")

    return x_tr, y_tr, a_tr, lost_tr, x_te, y_te, a_te, lost_te


def _choice_logprob(p1: float, action: float, lapse_rate: float) -> float:
    lam = float(np.clip(lapse_rate, 1e-9, 1.0 - 1e-9))
    p_model = p1 if action > 0.5 else (1.0 - p1)
    p = (1.0 - lam) * p_model + lam * 0.5
    return float(np.log(np.clip(p, 1e-12, 1.0)))


def _inverse_distance_scores(
    x: np.ndarray,
    mem_x: np.ndarray,
    mem_y: np.ndarray,
    weights: np.ndarray,
    distance_eps: float,
    base_mass: float,
) -> Tuple[float, float]:
    if mem_x.shape[0] == 0:
        return base_mass, base_mass

    d = np.linalg.norm(mem_x - x[None, :], axis=1)
    inv_d = 1.0 / (d + distance_eps)
    weighted = weights * inv_d

    s0 = base_mass + float(np.sum(weighted[mem_y == 0]))
    s1 = base_mass + float(np.sum(weighted[mem_y == 1]))
    return s0, s1


def _scores_to_p1(s0: float, s1: float) -> float:
    z = s0 + s1
    if not np.isfinite(z) or z <= 0:
        return 0.5
    return float(np.clip(s1 / z, 1e-9, 1.0 - 1e-9))


@dataclass
class LastMInverseDistanceModel:
    """
    Working-memory model where decisions are based only on the last M
    feedback-available observations.
    """

    M: int = 20
    use_category_specific_memory: bool = False
    M0: int = 20
    M1: int = 20
    distance_eps: float = 1e-6
    base_mass: float = 1.0

    def _select_memory(
        self,
        mem_x: np.ndarray,
        mem_y: np.ndarray,
        M: int,
        use_category_specific_memory: bool,
        M0: int,
        M1: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if mem_x.shape[0] == 0:
            return mem_x, mem_y

        if not use_category_specific_memory:
            M_use = max(int(M), 1)
            if mem_x.shape[0] > M_use:
                return mem_x[-M_use:], mem_y[-M_use:]
            return mem_x, mem_y

        m0 = max(int(M0), 0)
        m1 = max(int(M1), 0)
        idx0 = np.where(mem_y == 0)[0]
        idx1 = np.where(mem_y == 1)[0]
        keep0 = idx0[-m0:] if m0 > 0 else np.array([], dtype=int)
        keep1 = idx1[-m1:] if m1 > 0 else np.array([], dtype=int)
        keep = np.concatenate([keep0, keep1])
        if keep.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)
        keep = np.sort(keep)
        return mem_x[keep], mem_y[keep]

    def _p1_from_memory(
        self,
        x: np.ndarray,
        mem_x: np.ndarray,
        mem_y: np.ndarray,
        M: int,
        use_category_specific_memory: bool,
        M0: int,
        M1: int,
    ) -> float:
        mem_x, mem_y = self._select_memory(
            mem_x=mem_x,
            mem_y=mem_y,
            M=M,
            use_category_specific_memory=use_category_specific_memory,
            M0=M0,
            M1=M1,
        )
        w = np.ones(mem_x.shape[0], dtype=float)
        s0, s1 = _inverse_distance_scores(
            x, mem_x, mem_y, w, self.distance_eps, self.base_mass
        )
        return _scores_to_p1(s0, s1)

    def log_likelihood(
        self,
        data: Dict[str, Any],
        lapse_rate: float = 0.0,
        M: Optional[int] = None,
        use_category_specific_memory: Optional[bool] = None,
        M0: Optional[int] = None,
        M1: Optional[int] = None,
    ) -> float:
        x_tr, y_tr, a_tr, lost_tr, x_te, _, a_te, lost_te = _prep(data)
        M_eff = self.M if M is None else int(M)
        use_cat = (
            self.use_category_specific_memory
            if use_category_specific_memory is None
            else bool(use_category_specific_memory)
        )
        M0_eff = self.M0 if M0 is None else int(M0)
        M1_eff = self.M1 if M1 is None else int(M1)

        mem_x = np.zeros((0, 2), dtype=float)
        mem_y = np.zeros((0,), dtype=int)

        ll = 0.0

        for x, y, a, lost in zip(x_tr, y_tr, a_tr, lost_tr):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x, mem_x, mem_y, M_eff, use_cat, M0_eff, M1_eff
                )
                ll += _choice_logprob(p1, a, lapse_rate)
                # Feedback available: add correct observation to memory.
                mem_x = np.vstack([mem_x, x[None, :]])
                mem_y = np.append(mem_y, y)

        for x, a, lost in zip(x_te, a_te, lost_te):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x, mem_x, mem_y, M_eff, use_cat, M0_eff, M1_eff
                )
                ll += _choice_logprob(p1, a, lapse_rate)

        return float(ll)

    def predict(
        self,
        data: Dict[str, Any],
        lapse_rate: float = 0.0,
        M: Optional[int] = None,
        use_category_specific_memory: Optional[bool] = None,
        M0: Optional[int] = None,
        M1: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        x_tr, y_tr, _, lost_tr, x_te, _, _, lost_te = _prep(data)
        M_eff = self.M if M is None else int(M)
        use_cat = (
            self.use_category_specific_memory
            if use_category_specific_memory is None
            else bool(use_category_specific_memory)
        )
        M0_eff = self.M0 if M0 is None else int(M0)
        M1_eff = self.M1 if M1 is None else int(M1)

        mem_x = np.zeros((0, 2), dtype=float)
        mem_y = np.zeros((0,), dtype=int)

        p1_tr = np.full(x_tr.shape[0], np.nan, dtype=float)
        p1_te = np.full(x_te.shape[0], np.nan, dtype=float)

        for t, (x, y, lost) in enumerate(zip(x_tr, y_tr, lost_tr)):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x, mem_x, mem_y, M_eff, use_cat, M0_eff, M1_eff
                )
                p1_tr[t] = (1.0 - lapse_rate) * p1 + lapse_rate * 0.5
                mem_x = np.vstack([mem_x, x[None, :]])
                mem_y = np.append(mem_y, y)

        for t, (x, lost) in enumerate(zip(x_te, lost_te)):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x, mem_x, mem_y, M_eff, use_cat, M0_eff, M1_eff
                )
                p1_te[t] = (1.0 - lapse_rate) * p1 + lapse_rate * 0.5

        return {"p1_train": p1_tr, "p1_test": p1_te}


@dataclass
class ExponentialDecayInverseDistanceModel:
    """
    Working-memory model where all past feedback-available observations contribute,
    with exponentially decaying recency weights exp(-decay_rate * age).
    """

    decay_rate: float = 0.1
    use_dual_decay: bool = False
    decay_time: float = 0.1
    decay_count: float = 0.1
    distance_eps: float = 1e-6
    base_mass: float = 1.0

    def _dual_decay_weights(
        self,
        mem_y: np.ndarray,
        decay_time: float,
        decay_count: float,
    ) -> np.ndarray:
        n = mem_y.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=float)
        d_t = max(float(decay_time), 0.0)
        d_c = max(float(decay_count), 0.0)
        t_abs = np.arange(n - 1, -1, -1, dtype=float)  # 0 for most recent
        n_same = np.zeros((n,), dtype=float)
        for i in range(n):
            if i + 1 < n:
                n_same[i] = float(np.sum(mem_y[i + 1 :] == mem_y[i]))
        return np.exp(-(d_t * t_abs + d_c * n_same))

    def _p1_from_memory(
        self,
        x: np.ndarray,
        mem_x: np.ndarray,
        mem_y: np.ndarray,
        decay_rate: float,
        use_dual_decay: bool,
        decay_time: float,
        decay_count: float,
    ) -> float:
        n = mem_x.shape[0]
        if n == 0:
            return 0.5

        if use_dual_decay:
            w = self._dual_decay_weights(mem_y, decay_time, decay_count)
        else:
            ages = np.arange(n - 1, -1, -1, dtype=float)  # 0 for most recent
            decay = max(float(decay_rate), 0.0)
            w = np.exp(-decay * ages)
        s0, s1 = _inverse_distance_scores(
            x, mem_x, mem_y, w, self.distance_eps, self.base_mass
        )
        return _scores_to_p1(s0, s1)

    def log_likelihood(
        self,
        data: Dict[str, Any],
        lapse_rate: float = 0.0,
        decay_rate: Optional[float] = None,
        use_dual_decay: Optional[bool] = None,
        decay_time: Optional[float] = None,
        decay_count: Optional[float] = None,
    ) -> float:
        x_tr, y_tr, a_tr, lost_tr, x_te, _, a_te, lost_te = _prep(data)
        d_eff = self.decay_rate if decay_rate is None else float(decay_rate)
        use_dual = (
            self.use_dual_decay if use_dual_decay is None else bool(use_dual_decay)
        )
        d_time = self.decay_time if decay_time is None else float(decay_time)
        d_count = self.decay_count if decay_count is None else float(decay_count)

        mem_x = np.zeros((0, 2), dtype=float)
        mem_y = np.zeros((0,), dtype=int)

        ll = 0.0

        for x, y, a, lost in zip(x_tr, y_tr, a_tr, lost_tr):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x,
                    mem_x,
                    mem_y,
                    d_eff,
                    use_dual,
                    d_time,
                    d_count,
                )
                ll += _choice_logprob(p1, a, lapse_rate)
                mem_x = np.vstack([mem_x, x[None, :]])
                mem_y = np.append(mem_y, y)

        for x, a, lost in zip(x_te, a_te, lost_te):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x,
                    mem_x,
                    mem_y,
                    d_eff,
                    use_dual,
                    d_time,
                    d_count,
                )
                ll += _choice_logprob(p1, a, lapse_rate)

        return float(ll)

    def predict(
        self,
        data: Dict[str, Any],
        lapse_rate: float = 0.0,
        decay_rate: Optional[float] = None,
        use_dual_decay: Optional[bool] = None,
        decay_time: Optional[float] = None,
        decay_count: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        x_tr, y_tr, _, lost_tr, x_te, _, _, lost_te = _prep(data)
        d_eff = self.decay_rate if decay_rate is None else float(decay_rate)
        use_dual = (
            self.use_dual_decay if use_dual_decay is None else bool(use_dual_decay)
        )
        d_time = self.decay_time if decay_time is None else float(decay_time)
        d_count = self.decay_count if decay_count is None else float(decay_count)

        mem_x = np.zeros((0, 2), dtype=float)
        mem_y = np.zeros((0,), dtype=int)

        p1_tr = np.full(x_tr.shape[0], np.nan, dtype=float)
        p1_te = np.full(x_te.shape[0], np.nan, dtype=float)

        for t, (x, y, lost) in enumerate(zip(x_tr, y_tr, lost_tr)):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x,
                    mem_x,
                    mem_y,
                    d_eff,
                    use_dual,
                    d_time,
                    d_count,
                )
                p1_tr[t] = (1.0 - lapse_rate) * p1 + lapse_rate * 0.5
                mem_x = np.vstack([mem_x, x[None, :]])
                mem_y = np.append(mem_y, y)

        for t, (x, lost) in enumerate(zip(x_te, lost_te)):
            if lost == 0:
                p1 = self._p1_from_memory(
                    x,
                    mem_x,
                    mem_y,
                    d_eff,
                    use_dual,
                    d_time,
                    d_count,
                )
                p1_te[t] = (1.0 - lapse_rate) * p1 + lapse_rate * 0.5

        return {"p1_train": p1_tr, "p1_test": p1_te}
