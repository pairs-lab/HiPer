"""
hiper.evaluation.clip_vocab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Open-vocabulary (CLIP zero-shot) labels over the AI2-THOR object types,
shared by the simulated (THOR-Lifelong) and the real-data (ObChange)
pipelines so both use the same model, prompts and vocabulary.

Labels are AI2-THOR TYPE KEYS ("AlarmClock"), so they index
config/class_priors_indoor.yaml directly; the text prompts use the
humanised form ("alarm clock").

Model: OpenAI CLIP ViT-B/32. In open_clip 3.3 the name 'ViT-B-32' with
pretrained='openai' builds GELU blocks, while the OpenAI weights were
trained with QuickGELU; 'ViT-B-32-quickgelu' is the matching architecture.

torch / open_clip are imported lazily; everything else is pure python.
"""

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hiper.evaluation.paths import THOR_TYPES_JSON as _TYPES

CLIP_MODEL = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_PROMPTS = ("a photo of a {}.", "a {} in a room.")
CLIP_LOGIT_SCALE = 100.0
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
THOR_TYPES_JSON = str(_TYPES)

_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def humanize(type_key: str) -> str:
    """CamelCase type key -> lowercase words: 'AlarmClock' -> 'alarm clock',
    'TVStand' -> 'tv stand', 'CD' -> 'cd' (same rule as
    lifelong_objects.load_class_lambdas)."""
    return _SPLIT.sub(" ", str(type_key)).lower()


def thor_types(path: str = THOR_TYPES_JSON,
               exclude: Sequence[str] = ()) -> List[str]:
    """Sorted AI2-THOR type keys from object_types.json ({'types': {...}})
    or from a plain JSON list."""
    with open(path) as f:
        d = json.load(f)
    keys = d["types"].keys() if isinstance(d, dict) else d
    return sorted(k for k in keys if k not in set(exclude))


def prompts_for(type_key: str,
                templates: Sequence[str] = CLIP_PROMPTS) -> List[str]:
    return [t.format(humanize(type_key)) for t in templates]


def label_probs(img_emb: np.ndarray, text_emb: np.ndarray,
                logit_scale: float = CLIP_LOGIT_SCALE) -> np.ndarray:
    """Softmax over the vocabulary of logit_scale * cosine, (N, T)."""
    a = np.asarray(img_emb, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    t = np.asarray(text_emb, dtype=np.float64)
    t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)
    z = logit_scale * (a @ t.T)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def aggregate_label(probs: np.ndarray, types: Sequence[str],
                    k: int = 5) -> Tuple[str, float, List[List]]:
    """Average per-frame softmax rows -> (label, its mean probability,
    top-k [[label, prob], ...])."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1, len(types)).mean(0)
    order = np.argsort(-p)[:k]
    top = [[str(types[i]), float(p[i])] for i in order]
    return top[0][0], top[0][1], top


def label_embeddings(img_emb: np.ndarray, text_emb: np.ndarray,
                     types: Sequence[str],
                     logit_scale: float = CLIP_LOGIT_SCALE
                     ) -> Tuple[str, float, List[List]]:
    """Zero-shot label of one object from its (F, d) per-view embeddings."""
    return aggregate_label(label_probs(img_emb, text_emb, logit_scale),
                           types)


class ClipVocab:
    """CLIP model + text embeddings of the vocabulary (prompt ensemble:
    mean of the normalised prompt embeddings, renormalised)."""

    def __init__(self, types: Optional[Sequence[str]] = None,
                 device: Optional[str] = None,
                 templates: Sequence[str] = CLIP_PROMPTS):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.types = list(types) if types is not None else thor_types()
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED)
        self.model = model.to(self.device).eval()
        tok = open_clip.get_tokenizer(CLIP_MODEL)
        texts = [p for t in self.types for p in prompts_for(t, templates)]
        with torch.no_grad():
            e = self.model.encode_text(tok(texts).to(self.device)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.reshape(len(self.types), len(templates), -1).mean(1)
        self.text_emb = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(
            1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def encode(self, images, batch: int = 256) -> np.ndarray:
        """images: float tensor (N, 3, 224, 224) in [0, 1] on any device ->
        (N, d) L2-normalised embeddings (numpy float32)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = images[i:i + batch].to(self.device, torch.float32)
                x = (x - self.mean) / self.std
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=self.device.startswith("cuda")):
                    e = self.model.encode_image(x)
                e = e.float()
                out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        if not out:
            return np.zeros((0, self.text_emb.shape[1]), np.float32)
        return torch.cat(out).numpy().astype(np.float32)

    def label(self, img_emb: np.ndarray) -> Tuple[str, float, List[List]]:
        return label_embeddings(img_emb, self.text_emb, self.types)

    def probs(self, img_emb: np.ndarray) -> np.ndarray:
        return label_probs(img_emb, self.text_emb)


def vocab_summary(types: Sequence[str]) -> Dict[str, str]:
    """{type key: humanised name} for reports."""
    return {t: humanize(t) for t in types}
