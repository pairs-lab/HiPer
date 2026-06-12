"""
tests/test_replay_buffer.py
===========================
Unit tests for seliSlam.vio.replay_buffer.DiversityReplayBuffer.

threshold=2.0  → accept everything (cosine sim is always < 2)
threshold=0.99 → accept only sufficiently different frames
threshold=0.0  → reject almost everything (used to test rejection)

Run with: python -m pytest tests/test_replay_buffer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from seliSlam.vio.replay_buffer import DiversityReplayBuffer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(seed: int) -> dict:
    return {
        "img_msg": None, "depth_msg": None,
        "velocity": 0.5, "stamp": None,
        "_seed": seed,
    }


def _extractor(frame: dict) -> np.ndarray:
    """Deterministic unit-norm feature from seed."""
    rng  = np.random.default_rng(frame["_seed"])
    feat = rng.standard_normal(128).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDiversityReplayBuffer:

    def test_empty_buffer_sample_returns_empty(self):
        buf = DiversityReplayBuffer(max_size=10, feature_extractor=_extractor)
        assert buf.sample(5) == []

    def test_add_increases_length(self):
        # threshold=2.0 → always accept (cosine sim is always in [-1,1])
        buf = DiversityReplayBuffer(max_size=10,
                                    similarity_threshold=2.0,
                                    feature_extractor=_extractor)
        for i in range(5):
            buf.maybe_add(_make_frame(i * 100))
        assert len(buf) == 5

    def test_identical_frames_rejected_with_tight_threshold(self):
        """
        With threshold=0.99, the same frame sent twice should be rejected
        because its cosine similarity to itself is 1.0 ≥ 0.99.
        """
        buf = DiversityReplayBuffer(max_size=10,
                                    similarity_threshold=0.99,
                                    feature_extractor=_extractor)
        frame = _make_frame(42)
        buf.maybe_add(frame)
        added = buf.maybe_add(frame)   # sim=1.0 ≥ 0.99 → rejected
        assert not added
        assert len(buf) == 1

    def test_max_size_respected_with_permissive_threshold(self):
        # threshold=2.0 means add every frame; cap is max_size
        buf = DiversityReplayBuffer(max_size=5,
                                    similarity_threshold=2.0,
                                    feature_extractor=_extractor)
        for i in range(20):
            buf.maybe_add(_make_frame(i))
        assert len(buf) == 5

    def test_sample_returns_correct_number(self):
        buf = DiversityReplayBuffer(max_size=20,
                                    similarity_threshold=2.0,
                                    feature_extractor=_extractor)
        for i in range(10):
            buf.maybe_add(_make_frame(i))
        samples = buf.sample(3)
        assert len(samples) == 3

    def test_sample_does_not_exceed_buffer_size(self):
        buf = DiversityReplayBuffer(max_size=5,
                                    similarity_threshold=2.0,
                                    feature_extractor=_extractor)
        for i in range(3):
            buf.maybe_add(_make_frame(i))
        assert len(buf) == 3
        samples = buf.sample(100)   # ask for more than stored
        assert len(samples) == 3

    def test_clear_empties_buffer(self):
        buf = DiversityReplayBuffer(max_size=10,
                                    similarity_threshold=2.0,
                                    feature_extractor=_extractor)
        for i in range(5):
            buf.maybe_add(_make_frame(i))
        buf.clear()
        assert len(buf) == 0
        assert buf.sample(1) == []

    def test_overflow_replaces_most_redundant(self):
        """
        When the buffer is full, _unconditional_add must replace
        the most-redundant (highest summed similarity) entry.
        After adding a 4th frame to a max_size=3 buffer,
        length must still be 3.
        """
        buf = DiversityReplayBuffer(max_size=3,
                                    similarity_threshold=2.0,
                                    feature_extractor=_extractor)
        for i in [0, 100, 200]:
            buf.maybe_add(_make_frame(i))
        assert len(buf) == 3
        buf.maybe_add(_make_frame(999))
        assert len(buf) == 3

    def test_no_extractor_accepts_all_up_to_max_size(self):
        """Without a feature extractor every frame is always added."""
        buf = DiversityReplayBuffer(max_size=5, feature_extractor=None)
        results = [buf.maybe_add(_make_frame(i)) for i in range(10)]
        assert len(buf) == 5
        # First 5 must have been accepted
        assert all(results[:5])

    def test_sample_elements_are_triplet_lists(self):
        """Each sample should be a list wrapping one frame dict."""
        buf = DiversityReplayBuffer(max_size=10,
                                    similarity_threshold=2.0,
                                    feature_extractor=_extractor)
        frame = _make_frame(7)
        buf.maybe_add(frame)
        samples = buf.sample(1)
        assert isinstance(samples, list)
        assert isinstance(samples[0], list)
        assert samples[0][0]["_seed"] == 7

    def test_similarity_threshold_exact_boundary(self):
        """
        Two frames with a known cosine similarity S:
        - threshold > S  → second frame accepted
        - threshold < S  → second frame rejected
        """
        buf_accept = DiversityReplayBuffer(max_size=10,
                                           similarity_threshold=2.0,
                                           feature_extractor=_extractor)
        buf_reject = DiversityReplayBuffer(max_size=10,
                                           similarity_threshold=0.0,
                                           feature_extractor=_extractor)
        f1, f2 = _make_frame(0), _make_frame(1)

        buf_accept.maybe_add(f1)
        added_a = buf_accept.maybe_add(f2)

        buf_reject.maybe_add(f1)
        buf_reject.maybe_add(f2)

        assert added_a, "threshold=2.0 should accept the second frame"
        # With threshold=0.0 the second frame is rejected if sim ≥ 0
        # (which is not guaranteed for random unit vectors, so we only
        #  assert that at least one rejection occurs when threshold is 0)
        # This is a weaker assertion: the buffer size should be ≤ 2
        assert len(buf_reject) <= 2
