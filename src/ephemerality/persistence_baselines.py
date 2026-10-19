"""
hiper.ephemerality.persistence_baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Baseline persistence models for the open-loop replay (paper E2). Each is a
PersistenceModel and consumes the same per-object event stream as the v2
filter:
  create(oid, label, t) followed by observe(oid, t, HIT, n_obs=n) in the
  session that maps the object, then exactly one observe(oid, t, evidence)
  per later session (HIT | MISS | NONE), or relocate(oid, t).

  DraftV1Model       the draft (v1) method as written
  ELiteScalarModel   ELite's scalar global ephemerality (Eq. 7), per object
  AlwaysPersistModel never forgets
  ClassRateModel     PF(lambda_c) with class rates learned on the train split
"""

from dataclasses import asdict
from typing import Dict

from hiper.ephemerality.global_bpf import GlobalEphemeralityBPF, bayes_fuse
from hiper.ephemerality.persistence_v2 import (
    HIT, MISS, NONE, HierarchicalPersistenceFilter, PersistenceModel)


class DraftV1Model(PersistenceModel):
    """
    The draft method exactly as LifelongEvaluator.end_session runs it:
    GlobalEphemeralityBPF updates plus the KLSparsifier pruning rules with
    the runner's thresholds. One observe per object and session:
      HIT        bpf.update(eps_l_hit): Bayes fusion (ELite Eq. 7), THEN
                 survival decay over the time since last seen, then
                 Beta += (1 - eps_hat, eps_hat); sessions_absent = 0;
                 n_obs += info["n_obs"] (frames, default 1).
      MISS, NONE bpf.mark_not_observed: b += 1 - exp(-lambda (t - last_seen))
                 (cumulative since last seen); sessions_absent += 1.
    Known behaviours kept on purpose: decay after fusion (a short-lived
    class re-observed every day drifts towards ephemeral), Beta evidence
    taken from its own estimate, cumulative absence evidence and no
    visibility model (out of view counts as absent). The caller's eps_l is
    ignored: the draft local BPF returns the constant 0.1 for every point,
    so HIT always fuses eps_l_hit = 0.1. The draft has no relocation: a
    re-identified object is a re-observation. Not replicated: the runner's
    float32 map save/load between sessions (relative effect ~1e-7).

    presence(t) = 1 - GlobalEphemeralityBPF.get_epsilon_g(t).
    Prune: (i) eps_g >= tau_g and Var[p] < tau_v; (ii) n_obs < min_obs and
    sessions_absent > 0; (iii) sessions_absent >= absent_max.
    Commit (draft paper): Var[p] < tau_v.
    """

    name = "draft_v1"

    def __init__(
        self,
        lambda_table: Dict[str, float],
        default_lambda: float,
        eps_l_hit: float = 0.1,
        tau_g: float = 0.95,
        tau_v: float = 0.02,
        min_obs: int = 5,
        absent_max: int = 3,
    ):
        self.bpf = GlobalEphemeralityBPF(default_lambda=float(default_lambda),
                                         class_priors=dict(lambda_table))
        self.eps_l_hit = float(eps_l_hit)
        self.tau_g = float(tau_g)
        self.tau_v = float(tau_v)
        self.min_obs = int(min_obs)
        self.absent_max = int(absent_max)
        self._n_obs: Dict[str, int] = {}
        self._absent: Dict[str, int] = {}

    def _state(self, oid: str):
        return self.bpf.get_all_states()[oid]

    def create(self, oid: str, class_label: str, t: float) -> None:
        # The state GlobalEphemeralityBPF.update creates for a new object.
        self.bpf.load_object_state(oid, epsilon_g=0.5, persistence_prob=0.5,
                                   last_seen=float(t),
                                   class_label=class_label,
                                   beta_a=1.0, beta_b=1.0)
        self._n_obs[oid] = 0
        self._absent[oid] = 0

    def observe(self, oid: str, t: float, evidence: str,
                eps_l: float = 0.0, **info) -> None:
        if evidence == HIT:
            self.bpf.update(oid, self._state(oid).class_label,
                            self.eps_l_hit, float(t))
            self._n_obs[oid] += int(info.get("n_obs", 1))
            self._absent[oid] = 0
        elif evidence in (MISS, NONE):
            self._absent[oid] += 1
            self.bpf.mark_not_observed(oid, float(t))
        else:
            raise ValueError(f"unknown evidence {evidence!r}")

    def relocate(self, oid: str, t: float, **info) -> None:
        self.observe(oid, t, HIT, **info)

    def presence(self, oid: str, t: float) -> float:
        cls = self._state(oid).class_label
        return 1.0 - self.bpf.get_epsilon_g(oid, cls, float(t))

    def rate_mean(self, oid: str) -> float:
        return self._state(oid).persistence_mean()

    def rate_var(self, oid: str) -> float:
        return self._state(oid).persistence_var()

    def should_prune(self, oid: str, t: float) -> bool:
        st = self._state(oid)
        if st.epsilon_g >= self.tau_g and st.persistence_var() < self.tau_v:
            return True
        if self._n_obs[oid] < self.min_obs and self._absent[oid] > 0:
            return True
        return self._absent[oid] >= self.absent_max

    def is_committed(self, oid: str, t: float) -> bool:
        return self.rate_var(oid) < self.tau_v

    def remove(self, oid: str) -> None:
        self.bpf._states.pop(oid, None)      # the BPF has no removal API
        self._n_obs.pop(oid, None)
        self._absent.pop(oid, None)

    def export(self, oid: str) -> dict:
        return dict(asdict(self._state(oid)), n_obs=self._n_obs[oid],
                    sessions_absent=self._absent[oid])

    def load(self, oid: str, state: dict) -> None:
        st = dict(state)
        self._n_obs[oid] = int(st.pop("n_obs"))
        self._absent[oid] = int(st.pop("sessions_absent"))
        self.bpf.load_object_state(oid, **st)


class ELiteScalarModel(PersistenceModel):
    """
    ELite's global ephemerality (Gil et al., ICRA 2025, Eq. 7) ported to
    objects: a scalar eps (init eps_init) fused by Bayes' rule with eps_hit
    on HIT and eps_miss on MISS; NONE leaves it unchanged; no time model.
    presence = 1 - eps; prune if eps >= tau_e (ELite's default threshold is
    0.5). ELite keeps beliefs per map point, so a relocated object starts
    afresh at its new location: eps = Bayes(eps_init, eps_hit).
    """

    name = "elite"

    def __init__(self, eps_hit: float = 0.1, eps_miss: float = 0.9,
                 eps_init: float = 0.5, tau_e: float = 0.5):
        self.eps_hit = float(eps_hit)
        self.eps_miss = float(eps_miss)
        self.eps_init = float(eps_init)
        self.tau_e = float(tau_e)
        self._eps: Dict[str, float] = {}

    def create(self, oid: str, class_label: str, t: float) -> None:
        self._eps[oid] = self.eps_init

    def observe(self, oid: str, t: float, evidence: str,
                eps_l: float = 0.0, **info) -> None:
        if evidence == HIT:
            self._eps[oid] = bayes_fuse(self._eps[oid], self.eps_hit)
        elif evidence == MISS:
            self._eps[oid] = bayes_fuse(self._eps[oid], self.eps_miss)
        elif evidence != NONE:
            raise ValueError(f"unknown evidence {evidence!r}")

    def relocate(self, oid: str, t: float, **info) -> None:
        if oid not in self._eps:
            raise KeyError(oid)
        self._eps[oid] = bayes_fuse(self.eps_init, self.eps_hit)

    def presence(self, oid: str, t: float) -> float:
        return 1.0 - self._eps[oid]

    def should_prune(self, oid: str, t: float) -> bool:
        return self._eps[oid] >= self.tau_e

    def remove(self, oid: str) -> None:
        self._eps.pop(oid, None)

    def export(self, oid: str) -> dict:
        return {"eps": self._eps[oid]}

    def load(self, oid: str, state: dict) -> None:
        self._eps[oid] = float(state["eps"])


class AlwaysPersistModel(PersistenceModel):
    """Every mapped object stays: presence 1 - 1e-6, never pruned."""

    name = "always_persist"
    PRESENCE = 1.0 - 1e-6

    def __init__(self):
        self._ids: set = set()

    def create(self, oid: str, class_label: str, t: float) -> None:
        self._ids.add(oid)

    def observe(self, oid: str, t: float, evidence: str,
                eps_l: float = 0.0, **info) -> None:
        if oid not in self._ids:
            raise KeyError(oid)
        if evidence not in (HIT, MISS, NONE):
            raise ValueError(f"unknown evidence {evidence!r}")

    def relocate(self, oid: str, t: float, **info) -> None:
        self.observe(oid, t, HIT)

    def presence(self, oid: str, t: float) -> float:
        if oid not in self._ids:
            raise KeyError(oid)
        return self.PRESENCE

    def should_prune(self, oid: str, t: float) -> bool:
        return False

    def remove(self, oid: str) -> None:
        self._ids.discard(oid)


class ClassRateModel(HierarchicalPersistenceFilter):
    """
    PF(lambda_c) with data-driven class rates: the fixed-rate limit
    (learn_rate=False) of the v2 filter, with the table and the pooled
    default_lambda returned by priors.learn_class_rates on the TRAIN split.
    Other keyword arguments (p_miss, p_false, tau_pi, ...) as in
    HierarchicalPersistenceFilter.
    """

    name = "class_rate"

    def __init__(self, lambda_table: Dict[str, float], default_lambda: float,
                 **kwargs):
        super().__init__(lambda_table=lambda_table,
                         default_lambda=default_lambda,
                         learn_rate=False, use_class_prior=True, **kwargs)
        self.name = "class_rate"
