"""
hiper.ephemerality.global_bpf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Global (cross-session) ephemerality via the class-conditioned
Bayesian Persistence Filter (BPF).

Extends LTVS-BPF (Deng et al. 2023) with semantic class priors:

  p_T(t | class) = λ_class · exp(−λ_class · t)

Where λ_class is a class-specific decay rate:
  - "tree", "building" → very small λ (persist for years)
  - "person", "bicycle" → large λ (disappear in seconds/minutes)

Global ephemerality update (ELite Eq. 7, extended):
  ε_g_new = (ε_g · ε_l) / (ε_g · ε_l + (1−ε_g)(1−ε_l))

Then modulated by class-conditioned survival of the persistence
probability p = 1 − ε:
  p ← p · exp(−λ_class · Δt)   ⇔   ε ← 1 − (1 − ε) · exp(−λ_class · Δt)

so ephemerality grows with elapsed time at a class-specific rate.
A "building" (λ ≈ 1e-9 s⁻¹) stays persistent for years without being
observed; a "person" (λ ≈ 2e-3 s⁻¹) becomes ephemeral within minutes.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


def bayes_fuse(eps_g: float, eps_l: float) -> float:
    """
    ELite Eq. (7):
      ε_g_new = (ε_g · ε_l) / (ε_g · ε_l + (1−ε_g)(1−ε_l))
    """
    num = eps_g * eps_l
    den = num + (1.0 - eps_g) * (1.0 - eps_l)
    return float(num / (den + 1e-8))


def fuse_and_decay(eps_g: float, eps_l: float,
                   lam: float, delta_t: float) -> float:
    """
    One step of the class-conditioned BPF, shared by the ephemerality
    node and the map manager:
      1. Bayesian fusion (ELite Eq. 7) of the global and local estimates;
      2. class-conditioned survival decay of the persistence probability,
         eps <- 1 - (1 - eps) * exp(-lam * delta_t), so ephemerality grows
         with time since the last observation.
    Returns the instantaneous estimate in [0, 1].
    """
    fused = bayes_fuse(eps_g, eps_l)
    survival = float(np.exp(-lam * max(0.0, delta_t)))
    return float(np.clip(1.0 - (1.0 - fused) * survival, 0.0, 1.0))


@dataclass
class _ObjectBPFState:
    epsilon_g:        float = 0.5
    persistence_prob: float = 0.5
    last_seen:        float = 0.0
    class_label:      str   = "unknown"
    # Second-order ephemerality: Beta(a, b) posterior over persistence prob.
    beta_a:           float = 1.0
    beta_b:           float = 1.0

    def persistence_mean(self) -> float:
        s = self.beta_a + self.beta_b
        return self.beta_a / s if s > 0 else 0.5

    def persistence_var(self) -> float:
        a, b = self.beta_a, self.beta_b
        s = a + b
        return (a * b / (s * s * (s + 1.0))) if s > 0 else 0.25


class GlobalEphemeralityBPF:
    """
    Manages per-object global ephemerality across sessions.

    Parameters
    ----------
    default_lambda : float
        Exponential decay rate for unknown classes.
    class_priors : dict
        Mapping class_label -> λ (decay rate).
    """

    def __init__(
        self,
        default_lambda: float = 0.01,
        class_priors: Optional[Dict[str, float]] = None,
        beta_concentration: float = 10.0,
    ):
        self._default_lambda = default_lambda
        self._class_priors: Dict[str, float] = class_priors or {}
        self._beta_kappa = beta_concentration

        # object_id -> _ObjectBPFState
        self._states: Dict[str, _ObjectBPFState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update_class_priors(self, priors: Dict[str, float]):
        """Merge in new class priors (called when LTM is loaded)."""
        self._class_priors.update(priors)

    def load_object_state(
        self,
        object_id: str,
        epsilon_g: float,
        persistence_prob: float,
        last_seen: float,
        class_label: str,
        beta_a: Optional[float] = None,
        beta_b: Optional[float] = None,
    ):
        """Restore a previously serialised object state.

        If the Beta parameters are supplied (newer maps) they are used
        directly; otherwise the posterior is seeded from the scalar
        epsilon_g with the configured pseudo-count, treating a loaded
        object as carrying accumulated rather than uninformative evidence.
        """
        if beta_a is None or beta_b is None:
            eps = float(np.clip(epsilon_g, 0.0, 1.0))
            beta_a = self._beta_kappa * (1.0 - eps)
            beta_b = self._beta_kappa * eps
        self._states[object_id] = _ObjectBPFState(
            epsilon_g=epsilon_g,
            persistence_prob=persistence_prob,
            last_seen=last_seen,
            class_label=class_label,
            beta_a=max(beta_a, 1e-3),
            beta_b=max(beta_b, 1e-3),
        )

    def get_epsilon_g(
        self,
        object_id: str,
        class_label: str,
        current_time: float,
    ) -> float:
        """
        Return current global ephemerality for an object,
        accounting for time elapsed since last observation
        (using the class-conditioned survival prior).
        """
        if object_id not in self._states:
            # Unknown object: initialise at maximum uncertainty
            self._states[object_id] = _ObjectBPFState(
                epsilon_g=0.5,
                persistence_prob=0.5,
                last_seen=current_time,
                class_label=class_label,
            )
        state = self._states[object_id]
        lam   = self._class_priors.get(class_label, self._default_lambda)

        # Class-conditioned survival of the persistence probability:
        # p decays as exp(-lambda dt), so ephemerality grows with absence.
        delta_t  = max(0.0, current_time - state.last_seen)
        survival = float(np.exp(-lam * delta_t))
        return float(np.clip(
            1.0 - (1.0 - state.epsilon_g) * survival, 0.0, 1.0))

    def update(
        self,
        object_id: str,
        class_label: str,
        epsilon_l: float,   # local ε from current session
        current_time: float,
    ) -> float:
        """
        Bayesian update of global ephemerality given a new
        local ephemerality observation.

        Returns the updated ε_g value.
        """
        if object_id not in self._states:
            self._states[object_id] = _ObjectBPFState(
                epsilon_g=0.5,
                persistence_prob=0.5,
                last_seen=current_time,
                class_label=class_label,
            )
        state = self._states[object_id]
        lam   = self._class_priors.get(class_label, self._default_lambda)

        # Fuse with the new local evidence, then apply the class-
        # conditioned survival decay -> instantaneous estimate.
        eg_hat = fuse_and_decay(state.epsilon_g, epsilon_l, lam,
                                current_time - state.last_seen)

        # Step 3: Beta posterior update (second-order ephemerality).
        # Persistence evidence is (1 - eg_hat); transience evidence is eg_hat.
        state.beta_a += (1.0 - eg_hat)
        state.beta_b += eg_hat

        # epsilon_g is the posterior mean of ephemerality, 1 - E[p].
        updated = float(1.0 - state.persistence_mean())

        # Write back
        state.epsilon_g        = updated
        state.persistence_prob = state.persistence_mean()
        state.last_seen        = current_time
        state.class_label      = class_label

        return updated

    def mark_not_observed(
        self,
        object_id: str,
        current_time: float,
    ):
        """
        Called for objects NOT seen in the current session.
        Adds class-conditioned transience evidence to the Beta posterior
        (absence is evidence of transience, never of persistence), then
        re-derives ε_g as the posterior mean 1 − E[p].
        """
        if object_id not in self._states:
            return
        state = self._states[object_id]
        lam   = self._class_priors.get(state.class_label,
                                        self._default_lambda)
        delta_t = max(0.0, current_time - state.last_seen)

        # Transience evidence grows with elapsed time at the class rate
        increase = 1.0 - float(np.exp(-lam * delta_t))
        state.beta_b += increase
        state.epsilon_g        = float(1.0 - state.persistence_mean())
        state.persistence_prob = state.persistence_mean()

    def get_all_states(self) -> Dict[str, _ObjectBPFState]:
        """Return a snapshot of all object states (for serialisation)."""
        return dict(self._states)

    def get_persistence_var(self, object_id: str) -> float:
        """Posterior variance of the persistence probability (confidence)."""
        if object_id not in self._states:
            return 0.25
        return float(self._states[object_id].persistence_var())

    def is_committed(self, object_id: str, tau_v: float) -> bool:
        """True once the Beta posterior is confident enough to act on."""
        return self.get_persistence_var(object_id) < tau_v

    # ── Static helpers ────────────────────────────────────────────────────────

    # Kept as a class attribute for API compatibility; the canonical
    # implementation is the module-level bayes_fuse().
    _bayes_fuse = staticmethod(bayes_fuse)
