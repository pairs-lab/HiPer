"""
hiper.ephemerality.persistence_pocd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Baseline: the POCD-style object stationarity belief of Bogenberger et al.,
"Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in
Real-World Semi-Static Environments", IEEE RA-L 11(3), 2026, ported from
github.com/learnsyslab/perceive_semantix_release @ 596a09c8 (MIT; the POCD
code is identical on its ``paper-version`` branch). The belief follows POCD
(Qian et al., RSS 2022).

Source (perceive_semantix_lib/core):
  scene_object.py    update_pocd_probabilities, updateKSingle, initial
                     belief Beta(2, 1) and N(0, 0.5)
  object_tracker.py  updated_measured_change (in view, not detected ->
                     30 m change), inject_fake_measured_change (decay)
  scene_belief.py    decay floor clamp, removal E[v] < 0.5
  config.py          ObjectMatchingConfig defaults, all kept

Per object: Beta(a, b) over the stationarity v, N(mu, sig) over the
geometric change, the previous inlier flag, the thresholded score c (= E[v],
raised to the decay floor after a pseudo-change), the static/dynamic prior
label, the last detection time and the last pseudo-change time.

One POCD step with measured change d (std 0.5 m): weight s from (prior
label, previous inlier flag): static 3/0, dynamic 2/0 (inlier/outlier);
C1 ~ k1 N(d; mu, sig) (unchanged), C2 ~ k2 U(|d|; 0, 10 m) (changed);
the posterior C1 Beta(a+s+1, b) + C2 Beta(a+s, b+1) is moment-matched and
scaled so that max(a, b) <= 10; N(mu, sig) is moment-matched likewise.

Their filter runs once per camera frame; the replayed stream has one event
per session. Each event is replayed as ``steps_per_session`` frames at the
session time:
  HIT       steps with d = 0 (re-observed in place; the stream carries no
            displacement; their d is the ICP translation)
  MISS      steps with d = 30 m (their in-view non-detection)
  relocate  one step with d = 30 m, then steps - 1 HIT steps: their
            same-frame path (old place seen empty, object re-found by the
            translate match); the belief is kept and the detection time is t
  NONE      no measurement; out-of-view decay only
Out-of-view decay (inject_fake_measured_change): while c > 0.6, a pseudo-
change d = 2 m (static and dynamic alike in their config) is injected once
per period (static 1800 s, dynamic 15 s) after the last detection, then
c = max(E[v], 0.6); decay never takes c below 0.6, so it never removes.
  idle_decay=True   continuous operation (their ROS node steps the scene
                    without data when idle): ticks every period between
                    events, and presence(t) includes the ticks up to t.
  idle_decay=False  the robot is off between sessions: one pseudo-change
                    per NONE event, none between sessions.
presence(t) = c (the quantity they threshold); prune: c < tau_remove (0.5).
Their LLM static/dynamic label is replaced by our class lifetime table:
1 / lambda_c >= 365 days -> static, else dynamic (unknown -> dynamic, their
default).
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

from hiper.ephemerality.persistence_v2 import (
    HIT,
    MISS,
    NONE,
    PersistenceModel,
)

# ObjectMatchingConfig defaults (config.py)
CHANGE_STD = 0.5     # pocd_default_change_meters_standard_deviation
MISS_CHANGE = 30.0   # pocd_default_change_meters_mean
DECAY_FLOOR = 0.6    # pocd_decay_stationarity_threshold
TAU_REMOVE = 0.5     # pocd_removal_threshold
# prior label -> (period s, pseudo-change mean m, std m); static, dynamic
DECAY: Dict[bool, Tuple[float, float, float]] = {
    True: (1800.0, 2.0, 0.5), False: (15.0, 2.0, 0.5)}
# SceneObject defaults (scene_object.py)
BETA0 = (2.0, 1.0)
GEO0 = (0.0, 0.5)
EPS, CAP = 1e-5, 10.0
# Safety bound on pseudo-changes per decay call; normally c reaches the
# floor in < 20 ticks, but an inlier pseudo-change can raise c instead.
MAX_DECAY_TICKS = 10000


@dataclass
class _POCDState:
    class_label: str
    static: bool
    t_obs: float                   # last detection
    t_fake: float                  # last pseudo-change (creation if none)
    a: float = BETA0[0]
    b: float = BETA0[1]
    mu: float = GEO0[0]
    sig: float = GEO0[1]
    inlier: bool = True
    conf: float = BETA0[0] / (BETA0[0] + BETA0[1])
    last_evidence: str = HIT


def pocd_step(st: _POCDState, change: float, std: float) -> None:
    """One SceneObject.update_pocd_probabilities call (in place)."""
    a, b, mu, sig = st.a, st.b, st.mu, st.sig
    s = (3.0 if st.static else 2.0) if st.inlier else 0.0
    tol = 20.0 * std
    s_sq = 1.0 / (1.0 / sig ** 2 + 1.0 / std ** 2)
    m = s_sq * (mu / sig ** 2 + change / std ** 2)
    # updateKSingle with Gamma(x + 1) = x Gamma(x); prior value 1 / 0
    ot = 1.0 if st.static else 0.0
    k1 = max(EPS, (a + s * ot) / (a + b + s))
    k2 = max(EPS, (b + s - s * ot) / (a + b + s))
    pdf = math.exp(-0.5 * ((change - mu) / sig) ** 2) / (
        sig * math.sqrt(2.0 * math.pi))
    C1 = max(EPS, k1 * max(pdf, EPS))
    C2 = max(EPS, k2 / tol)            # U(min(|d|, tol); 0, tol) = 1 / tol
    C1, C2 = C1 / (C1 + C2), C2 / (C1 + C2)
    mu_p = C1 * m + C2 * mu
    # = C1 (s_sq + m^2) + C2 (sig^2 + mu^2) - mu_p^2, cancellation-free
    var = C1 * s_sq + C2 * sig ** 2 + C1 * C2 * (m - mu) ** 2
    # Beta moments; their code fixes object_type = 1 here
    gamma = (a + s + 1.0) / (a + b + s + 1.0)
    eta = (a + s) / (a + b + s + 1.0)
    theta = C1 * gamma + C2 * eta
    den2 = (a + b + s + 1.0) * (a + b + s + 2.0)
    alpha = (a + s + 2.0) * (a + s + 1.0) / den2
    beta = (a + s + 1.0) * (a + s) / den2
    num = C1 * theta * alpha + C2 * beta * theta - theta ** 2
    den = theta ** 2 - C1 * alpha - C2 * beta
    a_n = num / den
    b_n = num * (1.0 - theta) / (den * theta)
    if a_n > CAP or b_n > CAP:
        r = max(a_n, b_n) / CAP
        a_n, b_n = a_n / r, b_n / r
    st.a, st.b, st.mu, st.sig = a_n, b_n, mu_p, math.sqrt(var)
    st.inlier = C1 >= C2
    st.conf = a_n / (a_n + b_n)


class POCDBetaModel(PersistenceModel):
    """Bogenberger et al. 2026 stationarity belief on the session stream."""

    name = "pocd"

    def __init__(
        self,
        lambda_table: Optional[Dict[str, float]] = None,
        default_lambda: float = 1e-6,
        static_days: float = 365.0,
        tau_remove: float = TAU_REMOVE,
        steps_per_session: int = 1,
        idle_decay: bool = True,
    ):
        self.lambda_table = dict(lambda_table or {})
        self.default_lambda = float(default_lambda)
        self.static_days = float(static_days)
        self.tau_remove = float(tau_remove)
        self.steps = max(1, int(steps_per_session))
        self.idle_decay = bool(idle_decay)
        self._s: Dict[str, _POCDState] = {}

    def is_static(self, class_label: str) -> bool:
        """Lifetime 1 / lambda_c >= static_days (relative tolerance 1e-9)."""
        lam = float(self.lambda_table.get(class_label, self.default_lambda))
        return lam * self.static_days * 86400.0 <= 1.0 + 1e-9

    # ── Decay ────────────────────────────────────────────────────────────────

    def _tick(self, st: _POCDState, tick: float) -> None:
        _, d, std = DECAY[st.static]
        pocd_step(st, d, std)
        st.conf = max(st.conf, DECAY_FLOOR)
        st.t_fake = tick

    def _decay(self, st: _POCDState, t: float) -> None:
        """Pseudo-changes of continuous operation strictly before t."""
        period = DECAY[st.static][0]
        for _ in range(MAX_DECAY_TICKS):
            tick = max(st.t_obs, st.t_fake) + period
            if st.conf <= DECAY_FLOOR or not tick < t:
                return
            self._tick(st, tick)

    # ── Interface ────────────────────────────────────────────────────────────

    def create(self, oid: str, class_label: str, t: float) -> None:
        self._s[oid] = _POCDState(class_label=class_label,
                                  static=self.is_static(class_label),
                                  t_obs=float(t), t_fake=float(t))

    def observe(self, oid: str, t: float, evidence: str,
                eps_l: float = 0.0, **info) -> None:
        st, t = self._s[oid], float(t)
        if self.idle_decay:
            self._decay(st, t)
        if evidence == HIT:
            for _ in range(self.steps):
                pocd_step(st, 0.0, CHANGE_STD)
            st.t_obs = t
        elif evidence == MISS:
            for _ in range(self.steps):
                pocd_step(st, MISS_CHANGE, CHANGE_STD)
        elif evidence == NONE:
            period = DECAY[st.static][0]
            if (not self.idle_decay and st.conf > DECAY_FLOOR
                    and t > max(st.t_obs, st.t_fake) + period):
                self._tick(st, t)
            return
        else:
            raise ValueError(f"unknown evidence {evidence!r}")
        st.last_evidence = evidence

    def relocate(self, oid: str, t: float) -> None:
        st, t = self._s[oid], float(t)
        if self.idle_decay:
            self._decay(st, t)
        pocd_step(st, MISS_CHANGE, CHANGE_STD)
        for _ in range(self.steps - 1):
            pocd_step(st, 0.0, CHANGE_STD)
        st.t_obs = t
        st.last_evidence = HIT

    def presence(self, oid: str, t: float) -> float:
        st = self._s[oid]
        if not self.idle_decay or float(t) <= st.t_obs:
            return st.conf
        tmp = _POCDState(**asdict(st))
        self._decay(tmp, float(t))
        return tmp.conf

    def rate_mean(self, oid: str) -> float:
        st = self._s[oid]
        return st.a / (st.a + st.b)

    def rate_var(self, oid: str) -> float:
        st = self._s[oid]
        s = st.a + st.b
        return st.a * st.b / (s * s * (s + 1.0))

    def should_prune(self, oid: str, t: float) -> bool:
        return self.presence(oid, t) < self.tau_remove

    def remove(self, oid: str) -> None:
        self._s.pop(oid, None)

    def export(self, oid: str) -> dict:
        return asdict(self._s[oid])

    def load(self, oid: str, state: dict) -> None:
        self._s[oid] = _POCDState(**state)

    def __contains__(self, oid: str) -> bool:
        return oid in self._s
