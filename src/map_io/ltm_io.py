"""
hiper.map_io.ltm_io
~~~~~~~~~~~~~~~~~~~~~~~~
Disk I/O for the long-term semantic map.

Uses numpy .npz format: lightweight, portable, no ROS dependency.
Each object's fields are stored as separate named arrays.
Class survival priors are stored as a JSON string.

File naming: <save_dir>/ltm_session_{session_id:04d}.npz
A symlink `ltm_latest.npz` always points to the most recent file.
"""

import glob
import json
import os
from typing import Dict, Optional

import numpy as np
import rospy

from hiper.map_io.object_record import ObjectRecord


class LTMIO:
    """
    Handles save/load of the long-term map in .npz format.

    Parameters
    ----------
    save_dir : str   Directory for map files.
    """

    def __init__(self, save_dir: str = "/tmp/hiper_maps"):
        self._save_dir = save_dir
        self.latest_path: Optional[str] = None
        os.makedirs(save_dir, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        objects: Dict[str, ObjectRecord],
        session_id: int,
        class_priors: dict,
        filepath: Optional[str] = None,
    ) -> str:
        """
        Serialise all ObjectRecords to a .npz file.
        Returns the absolute path of the saved file.
        """
        if filepath is None:
            filepath = self.default_path(session_id)

        arrays = {}
        obj_list = list(objects.values())
        n = len(obj_list)

        if n == 0:
            np.savez_compressed(filepath,
                                num_objects=np.int32(0),
                                class_priors_json=np.bytes_(
                                    json.dumps(class_priors)))
            self._update_symlink(filepath)
            self.latest_path = filepath
            return filepath

        # Stack each field into a 1-D or 2-D array
        arrays["num_objects"] = np.int32(n)
        arrays["class_priors_json"] = np.bytes_(json.dumps(class_priors))
        arrays["session_id"] = np.int32(session_id)

        # Per-object arrays
        arrays["object_ids"]   = np.array([o.object_id.encode()
                                            for o in obj_list], dtype=object)
        arrays["class_labels"] = np.array([o.class_label.encode()
                                            for o in obj_list], dtype=object)
        arrays["positions"]    = np.stack(
            [o.position for o in obj_list])             # (N, 3)
        arrays["axes"]         = np.stack(
            [o.axes for o in obj_list])                 # (N, 3)
        arrays["orientations"] = np.stack(
            [o.orientation for o in obj_list])          # (N, 4)
        arrays["position_covariances"] = np.stack(
            [o.position_covariance for o in obj_list])  # (N, 6)

        # Embeddings: pad to same length
        embed_dim = max(
            (o.embedding.shape[0] for o in obj_list
             if o.embedding is not None), default=0)
        if embed_dim > 0:
            embs = np.zeros((n, embed_dim), dtype=np.float32)
            for i, o in enumerate(obj_list):
                if o.embedding is not None:
                    embs[i] = o.embedding
            arrays["embeddings"] = embs
        else:
            arrays["embeddings"] = np.zeros((n, 0), dtype=np.float32)

        arrays["epsilon_g"]         = np.array(
            [o.epsilon_g for o in obj_list], dtype=np.float32)
        arrays["epsilon_l"]         = np.array(
            [o.epsilon_l for o in obj_list], dtype=np.float32)
        arrays["persistence_probs"] = np.array(
            [o.persistence_prob for o in obj_list], dtype=np.float32)

        # Unified belief state (theta_j): Beta posterior, scalar embedding
        # uncertainty, and the Welford running statistics that let u_j keep
        # accumulating across sessions (zero-length mean/m2 = not started).
        arrays["beta_a"] = np.array(
            [o.beta_a for o in obj_list], dtype=np.float32)
        arrays["beta_b"] = np.array(
            [o.beta_b for o in obj_list], dtype=np.float32)
        arrays["embedding_uncertainties"] = np.array(
            [o.embedding_uncertainty for o in obj_list], dtype=np.float32)
        welford_dim = max(
            (o._emb_mean.shape[0] for o in obj_list
             if o._emb_mean is not None), default=0)
        emb_means  = np.zeros((n, welford_dim), dtype=np.float64)
        emb_m2s    = np.zeros((n, welford_dim), dtype=np.float64)
        emb_counts = np.zeros(n, dtype=np.int32)
        for i, o in enumerate(obj_list):
            if o._emb_mean is not None and welford_dim > 0:
                d = o._emb_mean.shape[0]
                emb_means[i, :d] = o._emb_mean
                emb_m2s[i, :d]   = o._emb_m2
                emb_counts[i]    = o._emb_count
        arrays["emb_means"]  = emb_means
        arrays["emb_m2s"]    = emb_m2s
        arrays["emb_counts"] = emb_counts

        arrays["observation_counts"] = np.array(
            [o.observation_count for o in obj_list], dtype=np.int32)
        arrays["first_seen"] = np.array(
            [o.first_seen for o in obj_list], dtype=np.float64)
        arrays["last_seen"]  = np.array(
            [o.last_seen  for o in obj_list], dtype=np.float64)
        arrays["sessions_absent"] = np.array(
            [o.sessions_absent for o in obj_list], dtype=np.int32)
        arrays["last_sparsify_counts"] = np.array(
            [o.last_sparsify_count for o in obj_list], dtype=np.int32)

        np.savez_compressed(filepath, **arrays)
        self._update_symlink(filepath)
        self.latest_path = filepath
        rospy.loginfo(f"[ltm_io] Saved {n} objects to {filepath}")
        return filepath

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, filepath: str) -> Optional[dict]:
        """
        Load a .npz map file.
        Returns {"objects": dict[str, ObjectRecord],
                 "class_priors": dict, "session_id": int}
        or None if the file does not exist.
        """
        if not os.path.exists(filepath):
            rospy.logwarn(f"[ltm_io] Map file not found: {filepath}")
            return None

        data = np.load(filepath, allow_pickle=True)
        n    = int(data["num_objects"])

        try:
            class_priors = json.loads(
                data["class_priors_json"].item().decode())
        except Exception:
            class_priors = {}

        session_id = int(data.get("session_id", 0))
        objects: Dict[str, ObjectRecord] = {}

        if n == 0:
            return {"objects": objects,
                    "class_priors": class_priors,
                    "session_id": session_id}

        oids   = data["object_ids"]
        labels = data["class_labels"]
        embed_arr = data["embeddings"]

        for i in range(n):
            rec = ObjectRecord()
            rec.object_id    = oids[i].decode() if isinstance(
                oids[i], bytes) else str(oids[i])
            rec.class_label  = labels[i].decode() if isinstance(
                labels[i], bytes) else str(labels[i])
            rec.position     = data["positions"][i].astype(np.float32)
            rec.axes         = data["axes"][i].astype(np.float32)
            rec.orientation  = data["orientations"][i].astype(np.float32)
            rec.position_covariance = data["position_covariances"][i].astype(
                np.float32)
            emb = embed_arr[i] if embed_arr.shape[1] > 0 else None
            rec.embedding    = emb.astype(np.float32) if emb is not None else None
            rec.epsilon_g    = float(data["epsilon_g"][i])
            rec.epsilon_l    = float(data["epsilon_l"][i])
            rec.persistence_prob  = float(data["persistence_probs"][i])
            # Unified belief state — absent in maps written before the
            # uncertainty upgrade; defaults keep old maps loadable.
            if "beta_a" in data:
                rec.beta_a = float(data["beta_a"][i])
                rec.beta_b = float(data["beta_b"][i])
            if "embedding_uncertainties" in data:
                rec.embedding_uncertainty = float(
                    data["embedding_uncertainties"][i])
            if "emb_counts" in data and int(data["emb_counts"][i]) > 0:
                rec._emb_mean  = data["emb_means"][i].astype(np.float64)
                rec._emb_m2    = data["emb_m2s"][i].astype(np.float64)
                rec._emb_count = int(data["emb_counts"][i])
            rec.observation_count = int(data["observation_counts"][i])
            rec.first_seen   = float(data["first_seen"][i])
            rec.last_seen    = float(data["last_seen"][i])
            if "sessions_absent" in data:
                rec.sessions_absent = int(data["sessions_absent"][i])
            if "last_sparsify_counts" in data:
                rec.last_sparsify_count = int(
                    data["last_sparsify_counts"][i])
            objects[rec.object_id] = rec

        rospy.loginfo(f"[ltm_io] Loaded {n} objects from {filepath}")
        return {"objects": objects,
                "class_priors": class_priors,
                "session_id": session_id}

    def load_latest(self) -> Optional[dict]:
        """Load the most recent map file in the save directory."""
        symlink = os.path.join(self._save_dir, "ltm_latest.npz")
        if os.path.exists(symlink):
            return self.load(symlink)

        # Fall back to highest session number
        pattern = os.path.join(self._save_dir, "ltm_session_*.npz")
        files   = sorted(glob.glob(pattern))
        if not files:
            return None
        return self.load(files[-1])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def default_path(self, session_id: int) -> str:
        return os.path.join(
            self._save_dir, f"ltm_session_{session_id:04d}.npz")

    def _update_symlink(self, target: str):
        symlink = os.path.join(self._save_dir, "ltm_latest.npz")
        if os.path.lexists(symlink):
            os.remove(symlink)
        os.symlink(os.path.abspath(target), symlink)
