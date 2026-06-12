"""
seliSlam.map_io.sparsifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~
KL-divergence map sparsification (ObVi-SLAM §V-B style).

For an independent Gaussian prior per object, the KL-sparse
approximation reduces to retaining each object's marginal
mean and covariance from the full posterior — exactly the
block-diagonal solution from ObVi-SLAM Eq. (17).

Additionally prunes objects with:
  - epsilon_g >= prune_threshold (highly ephemeral, no longer useful)
  - observation_count < min_observations
"""

from typing import Dict
import numpy as np
from seliSlam.map_io.object_record import ObjectRecord


class KLSparsifier:
    """
    Parameters
    ----------
    kl_budget      : float   KL divergence budget per object (not strictly
                              enforced in the diagonal case — kept for API
                              compatibility with future dense prior support).
    prune_eg_threshold : float   Objects with ε_g above this are pruned.
    min_observations   : int     Objects seen fewer times are pruned.
    """

    def __init__(
        self,
        kl_budget: float = 0.01,
        prune_eg_threshold: float = 0.95,
        min_observations: int = 3,
        prune_var_threshold: float = 0.02,
        absent_sessions_threshold: int = 3,
    ):
        self._kl_budget  = kl_budget
        self._prune_eg   = prune_eg_threshold
        self._min_obs    = min_observations
        self._prune_var  = prune_var_threshold
        self._absent_max = absent_sessions_threshold

    def sparsify(
        self,
        objects: Dict[str, ObjectRecord],
    ) -> Dict[str, ObjectRecord]:
        """
        Apply sparsification and pruning.
        Returns a new dict containing only the retained objects.

        The "sparsification" step in the diagonal case is trivial:
        each object's Gaussian prior is already independent, so
        the KL-optimal sparse map is the current set of marginals.
        What we DO here is:
          1. Prune highly ephemeral objects, but only once the Beta
             posterior is confident (Var[p] < prune_var_threshold);
             objects with high mean ephemerality but still-wide variance
             are retained as tentative to avoid premature deletion.
          2. Prune under-observed objects (with a grace period for objects
             first seen this session, which have had no chance to
             accumulate observations yet).
          3. Prune objects absent for absent_sessions_threshold
             consecutive sessions.
          4. Shrink the block-diagonal prior covariance (position and
             embedding) for well-observed objects, encoding information gain.
        """
        retained: Dict[str, ObjectRecord] = {}

        for oid, obj in objects.items():
            # Pruning rule 1: too ephemeral AND confident about it
            if obj.epsilon_g >= self._prune_eg and \
                    obj.persistence_var() < self._prune_var:
                continue
            # Pruning rule 2: too few observations. Objects created this
            # session (never absent) get a grace period of one session.
            if obj.observation_count < self._min_obs and \
                    obj.sessions_absent > 0:
                continue
            # Pruning rule 3: absent too many consecutive sessions
            if obj.sessions_absent >= self._absent_max:
                continue

            # Covariance update: shrink uncertainty with observations.
            # The target prior is sigma^2_0 / sqrt(n); the incremental
            # factor sqrt(n_prev / n) reaches that target without
            # compounding when sparsify runs once per session.
            n = max(obj.observation_count, 1)
            n_prev = max(obj.last_sparsify_count, 1)
            if n > n_prev:
                shrink = float(np.sqrt(n_prev / float(n)))
                obj.position_covariance = (
                    obj.position_covariance * shrink).astype(np.float32)
                obj.embedding_uncertainty = float(
                    obj.embedding_uncertainty * shrink)
                obj.last_sparsify_count = n

            retained[oid] = obj

        return retained
