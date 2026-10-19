"""
hiper.ephemerality.persistence_v2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Object-level persistence beliefs for lifelong maps.

Per mapped object j the belief holds
  pi_j     P(object is at its mapped location) at time t_j of the last
           informative update,
  (a_j,b_j) Beta belief over p_j, the probability of surviving one
           reference gap dt_ref at the mapped location. Survival over a
           gap dt is p_j ** (dt / dt_ref), i.e. exponential survival with
           rate lambda_j = -ln(p_j) / dt_ref.
The class prior comes from the open-vocabulary (CLIP) label:
  E[p_j] = m_c = exp(-lambda_c * dt_ref),  a_j + b_j = kappa0.

Session update, predict then update (detection model of Rosen et al.,
"Towards lifelong feature-based mapping in semi-static environments",
ICRA 2016):
  r   = (t - t_j) / dt_ref,   S = E[p^r] = B(a + r, b) / B(a, b)
  pi- = pi S + (1 - pi) rho
  HIT : L1 = 1 - P_M, L0 = P_F      MISS: L1 = P_M, L0 = 1 - P_F
  NONE (not in view, or tentative match): no update, t_j kept.
  pi+ = pi- L1 / (pi- L1 + (1 - pi-) L0)
  Two-slice posterior over (present at t_j, present at t):
    w11 ~ pi S L1,  w10 ~ pi (1-S) L0,  w01 ~ (1-pi) rho L1,
    w00 ~ (1-pi)(1-rho) L0   (normalised)
  a += r * w11   (exact: survival likelihood p^r is conjugate)
  b += w10       (first order in (1 - p) for a vanish inside the gap)

With learn_rate=False the survival probability is the fixed class value
m_c and the filter is exactly Rosen's exponential persistence filter
PF(lambda_c); with use_class_prior=False every class uses default_lambda.

Decisions
  prune : pi(t) < tau_pi and (last evidence was MISS or Var[p] < tau_v);
          a belief that dropped only through prior decay is kept until
          the rate itself is confident (variance gate).
  commit: E[p] >= tau_commit and Var[p] < tau_v and pi(t) >= 0.5.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, Optional

HIT, MISS, NONE = "hit", "miss", "none"


class PersistenceModel(ABC):
    """Common interface of every persistence belief compared in the paper."""

    name: str = "base"

    @abstractmethod
    def create(self, oid: str, class_label: str, t: float) -> None:
        """Start tracking a newly mapped object first seen at time t."""

    @abstractmethod
    def observe(self, oid: str, t: float, evidence: str,
                eps_l: float = 0.0, **info) -> None:
        """Apply one session's evidence (HIT, MISS or NONE) at time t.
        ``info`` carries optional per-session extras (e.g. n_obs frames)."""

    def relocate(self, oid: str, t: float) -> None:
        """A confident re-identification found the object elsewhere."""
        self.observe(oid, t, MISS)
        self.observe(oid, t, HIT)

    @abstractmethod
    def presence(self, oid: str, t: float) -> float:
        """P(object is at its mapped location at time t)."""

    def rate_mean(self, oid: str) -> float:
        return float("nan")

    def rate_var(self, oid: str) -> float:
        return 0.0

    @abstractmethod
    def should_prune(self, oid: str, t: float) -> bool:
        """Map-maintenance decision at the end of a session."""

    def is_committed(self, oid: str, t: float) -> bool:
        return False

    @abstractmethod
    def remove(self, oid: str) -> None:
        """Forget an object (pruned or merged)."""

    def export(self, oid: str) -> dict:
        return {}

    def load(self, oid: str, state: dict) -> None:
        raise NotImplementedError


@dataclass
class _HPFState:
    class_label: str
    pi: float
    a: float
    b: float
    t: float
    last_evidence: str = HIT


def _log_beta_moment(a: float, b: float, r: float) -> float:
    """ln E[p^r] for p ~ Beta(a, b) and r >= 0."""
    if r <= 0.0:
        return 0.0
    return (math.lgamma(a + r) + math.lgamma(a + b)
            - math.lgamma(a) - math.lgamma(a + b + r))


class HierarchicalPersistenceFilter(PersistenceModel):
    """Class-conditioned persistence filter with a per-object Beta rate."""

    name = "hpf"

    def __init__(
        self,
        lambda_table: Optional[Dict[str, float]] = None,
        default_lambda: float = 1e-6,
        dt_ref: float = 86400.0,
        kappa0: float = 4.0,
        p_miss: float = 0.1,
        p_false: float = 0.05,
        rho: float = 0.0,
        pi_new: float = 1.0,
        tau_pi: float = 0.1,
        tau_v: float = 0.01,
        tau_commit: float = 0.9,
        learn_rate: bool = True,
        use_class_prior: bool = True,
        variance_gate: bool = True,
        m_clip: float = 1e-4,
        relocation_resets_rate: bool = False,
    ):
        self.lambda_table = dict(lambda_table or {})
        self.default_lambda = float(default_lambda)
        self.dt_ref = float(dt_ref)
        self.kappa0 = float(kappa0)
        self.p_miss = float(p_miss)
        self.p_false = float(p_false)
        self.rho = float(rho)
        self.pi_new = float(pi_new)
        self.tau_pi = float(tau_pi)
        self.tau_v = float(tau_v)
        self.tau_commit = float(tau_commit)
        self.learn_rate = bool(learn_rate)
        self.use_class_prior = bool(use_class_prior)
        self.variance_gate = bool(variance_gate)
        self.m_clip = float(m_clip)
        # Ablation: treat a relocated object as a new feature (location-
        # based persistence) instead of carrying its rate belief along.
        self.relocation_resets_rate = bool(relocation_resets_rate)
        self._s: Dict[str, _HPFState] = {}
        if not learn_rate:
            self.name = "pf_class" if use_class_prior else "pf"

    # ── Prior ────────────────────────────────────────────────────────────────

    def class_lambda(self, class_label: str) -> float:
        if not self.use_class_prior:
            return self.default_lambda
        return float(self.lambda_table.get(class_label, self.default_lambda))

    def prior_mean(self, class_label: str) -> float:
        m = math.exp(-self.class_lambda(class_label) * self.dt_ref)
        return min(max(m, self.m_clip), 1.0 - self.m_clip)

    # ── Survival over a gap ──────────────────────────────────────────────────

    def _survival(self, st: _HPFState, r: float) -> float:
        if r <= 0.0:
            return 1.0
        if not self.learn_rate:
            return self.prior_mean(st.class_label) ** r
        return math.exp(_log_beta_moment(st.a, st.b, r))

    # ── Interface ────────────────────────────────────────────────────────────

    def create(self, oid: str, class_label: str, t: float) -> None:
        m = self.prior_mean(class_label)
        self._s[oid] = _HPFState(class_label=class_label, pi=self.pi_new,
                                 a=self.kappa0 * m,
                                 b=self.kappa0 * (1.0 - m), t=float(t))

    def observe(self, oid: str, t: float, evidence: str,
                eps_l: float = 0.0, **info) -> None:
        st = self._s[oid]
        if evidence == NONE:
            return
        r = max(0.0, (float(t) - st.t) / self.dt_ref)
        S = self._survival(st, r)
        if evidence == HIT:
            L1, L0 = 1.0 - self.p_miss, self.p_false
            w = 1.0 - min(max(float(eps_l), 0.0), 1.0)
            L1, L0 = w * L1 + (1.0 - w), w * L0 + (1.0 - w)
        elif evidence == MISS:
            L1, L0 = self.p_miss, 1.0 - self.p_false
        else:
            raise ValueError(f"unknown evidence {evidence!r}")
        pi = st.pi
        w11 = pi * S * L1
        w10 = pi * (1.0 - S) * L0
        w01 = (1.0 - pi) * self.rho * L1
        w00 = (1.0 - pi) * (1.0 - self.rho) * L0
        z = w11 + w10 + w01 + w00
        if z <= 0.0:
            return
        w11, w10, w01, w00 = w11 / z, w10 / z, w01 / z, w00 / z
        if self.learn_rate:
            st.a += r * w11
            st.b += w10
        st.pi = w11 + w01
        st.t = float(t)
        st.last_evidence = evidence

    def relocate(self, oid: str, t: float) -> None:
        """The object left its mapped location and was re-identified at a
        new one: count a vanish event, then restart presence there."""
        st = self._s[oid]
        if self.relocation_resets_rate:
            m = self.prior_mean(st.class_label)
            st.a, st.b = self.kappa0 * m, self.kappa0 * (1.0 - m)
        elif self.learn_rate:
            st.b += st.pi
        st.pi = self.pi_new
        st.t = float(t)
        st.last_evidence = HIT

    def presence(self, oid: str, t: float) -> float:
        st = self._s[oid]
        r = max(0.0, (float(t) - st.t) / self.dt_ref)
        return st.pi * self._survival(st, r) + (1.0 - st.pi) * self.rho

    def rate_mean(self, oid: str) -> float:
        st = self._s[oid]
        if not self.learn_rate:
            return self.prior_mean(st.class_label)
        return st.a / (st.a + st.b)

    def rate_var(self, oid: str) -> float:
        if not self.learn_rate:
            return 0.0
        st = self._s[oid]
        s = st.a + st.b
        return st.a * st.b / (s * s * (s + 1.0))

    def should_prune(self, oid: str, t: float) -> bool:
        st = self._s[oid]
        if self.presence(oid, t) >= self.tau_pi:
            return False
        if st.last_evidence == MISS or not self.variance_gate:
            return True
        return self.rate_var(oid) < self.tau_v

    def is_committed(self, oid: str, t: float) -> bool:
        gate = (self.rate_var(oid) < self.tau_v) if self.variance_gate \
            else True
        return (self.rate_mean(oid) >= self.tau_commit and gate
                and self.presence(oid, t) >= 0.5)

    def remove(self, oid: str) -> None:
        self._s.pop(oid, None)

    def export(self, oid: str) -> dict:
        return asdict(self._s[oid])

    def load(self, oid: str, state: dict) -> None:
        self._s[oid] = _HPFState(**state)

    def __contains__(self, oid: str) -> bool:
        return oid in self._s
