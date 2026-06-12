"""
seliSlam.segmentation.mobile_sam_wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MobileSAM (Zhang et al. 2023) + CLIP (or DINOv2) for open-vocabulary
instance segmentation.

Pipeline per frame:
  1. MobileSAM automatic mask generation → N binary masks
  2. Crop each masked region from the RGB image
  3. Pass crops through CLIP / DINOv2 → per-instance embedding
  4. Zero-shot class label via cosine similarity against a text prompt bank

The class label bank is built from CLIP text embeddings at __init__ time
so inference is fast (no text encoding at runtime).

Install prerequisites:
  pip install mobile_sam clip    (or pip install git+https://...)
  pip install open_clip_torch    (alternative CLIP implementation)

If mobile_sam is unavailable, falls back to a grid-of-squares segmentation
(useful for development / testing without GPU).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Class label bank (extend with your scene-specific classes) ────────────────
_DEFAULT_CLASSES = [
    "tree", "building", "person", "car", "bicycle",
    "bench", "lamppost", "trash can", "table", "chair",
    "wall", "fence", "road", "grass", "sky",
    "door", "window", "sign", "bus", "motorcycle",
]


class MobileSAMWithCLIP:
    """
    Open-vocabulary segmenter.

    Parameters
    ----------
    clip_model : str   CLIP model variant, e.g. "ViT-B/32"
    device     : str   "cuda" or "cpu"
    class_bank : list  String labels for zero-shot classification.
                       Defaults to _DEFAULT_CLASSES.
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        device: str = "cuda",
        class_bank: list = None,
        confidence_threshold: float = 0.5,
        max_instances: int = 32,
    ):
        self._device    = device
        self._conf      = confidence_threshold
        self._max_inst  = max_instances
        self._classes   = class_bank or _DEFAULT_CLASSES

        self._sam    = None
        self._clip   = None
        self._clip_preprocess = None
        self._text_feats = None   # (C, embed_dim) pre-computed text embeddings

        self._load_models(clip_model)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self, clip_model: str):
        try:
            self._load_sam()
        except Exception as e:
            logger.warning(f"[mobile_sam_wrapper] MobileSAM unavailable: {e}. "
                           "Using grid fallback.")

        try:
            self._load_clip(clip_model)
        except Exception as e:
            logger.warning(f"[mobile_sam_wrapper] CLIP unavailable: {e}. "
                           "Embeddings will be zeros.")

    def _load_sam(self):
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        import os
        import urllib.request

        ckpt_path = os.path.expanduser(
            "~/.seliSlam/checkpoints/mobile_sam.pt")
        if not os.path.exists(ckpt_path):
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            logger.info("[mobile_sam_wrapper] Downloading MobileSAM weights...")
            urllib.request.urlretrieve(
                "https://github.com/ChaoningZhang/MobileSAM/raw/master/"
                "weights/mobile_sam.pt",
                ckpt_path)

        sam = sam_model_registry["vit_t"](checkpoint=ckpt_path)
        sam.to(self._device).eval()
        self._sam = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,          # fewer points = faster on Jetson
            pred_iou_thresh=self._conf,
            stability_score_thresh=0.85,
            min_mask_region_area=200,
        )
        logger.info("[mobile_sam_wrapper] MobileSAM loaded.")

    def _load_clip(self, model_name: str):
        import torch
        try:
            import clip as openai_clip
            model, preprocess = openai_clip.load(
                model_name, device=self._device)
            self._clip_lib     = "openai"
            self._clip         = model
            self._clip_preprocess = preprocess
            self._embed_dim    = model.visual.output_dim
            # Pre-compute text embeddings for each class label
            with torch.no_grad():
                texts = openai_clip.tokenize(
                    [f"a photo of a {c}" for c in self._classes]
                ).to(self._device)
                self._text_feats = model.encode_text(texts).float()
                self._text_feats /= self._text_feats.norm(
                    dim=1, keepdim=True) + 1e-8
        except ImportError:
            # Fall back to open_clip
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self._device)
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._clip_lib     = "openclip"
            self._clip         = model
            self._clip_preprocess = preprocess
            self._embed_dim    = 512
            with torch.no_grad():
                texts = tokenizer(
                    [f"a photo of a {c}" for c in self._classes]
                ).to(self._device)
                self._text_feats = model.encode_text(texts).float()
                self._text_feats /= self._text_feats.norm(
                    dim=1, keepdim=True) + 1e-8

        logger.info(f"[mobile_sam_wrapper] CLIP ({self._clip_lib}) "
                    f"embed_dim={self._embed_dim} loaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def segment(self, rgb_img: np.ndarray) -> tuple:
        """
        Run segmentation on a (H, W, 3) uint8 RGB image.

        Returns
        -------
        masks_list    : list of (H, W) bool arrays
        scores        : list of float confidence scores
        class_labels  : list of str class labels
        embeddings    : list of (embed_dim,) float32 arrays
        """
        if self._sam is None:
            return self._grid_fallback(rgb_img)

        # Step 1: automatic mask generation
        sam_results = self._sam.generate(rgb_img)
        sam_results = sorted(sam_results,
                             key=lambda x: x["area"], reverse=True)
        sam_results = sam_results[:self._max_inst]

        if len(sam_results) == 0:
            return [], [], [], []

        masks_list   = [r["segmentation"] for r in sam_results]
        scores       = [float(r["predicted_iou"]) for r in sam_results]
        embeddings   = []
        class_labels = []

        for mask in masks_list:
            emb, label = self._embed_and_classify(rgb_img, mask)
            embeddings.append(emb)
            class_labels.append(label)

        return masks_list, scores, class_labels, embeddings

    def _embed_and_classify(
        self,
        rgb_img: np.ndarray,
        mask:    np.ndarray,
    ) -> tuple:
        """
        Crop the masked region from the image and pass it through CLIP.
        Returns (embedding: np.ndarray, class_label: str).
        """
        import torch
        from PIL import Image as PILImage

        if self._clip is None:
            dim = getattr(self, "_embed_dim", 512)
            return np.zeros(dim, dtype=np.float32), "unknown"

        # Bounding box crop of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return np.zeros(self._embed_dim, dtype=np.float32), "unknown"

        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        # Pad crop slightly
        pad = 8
        r0 = max(0, r0 - pad)
        r1 = min(rgb_img.shape[0]-1, r1 + pad)
        c0 = max(0, c0 - pad)
        c1 = min(rgb_img.shape[1]-1, c1 + pad)

        crop = rgb_img[r0:r1+1, c0:c1+1]
        if crop.size == 0:
            return np.zeros(self._embed_dim, dtype=np.float32), "unknown"

        pil_crop = PILImage.fromarray(crop)

        # CLIP encode
        with torch.no_grad():
            img_input = self._clip_preprocess(pil_crop).unsqueeze(0).to(
                self._device)
            if self._clip_lib == "openai":
                img_feat = self._clip.encode_image(img_input).float()
            else:
                img_feat = self._clip.encode_image(img_input).float()

            img_feat = img_feat / (img_feat.norm(dim=1, keepdim=True) + 1e-8)
            emb_np   = img_feat.squeeze(0).cpu().numpy().astype(np.float32)

            # Zero-shot classification
            if self._text_feats is not None:
                sims  = (img_feat @ self._text_feats.T).squeeze(0)
                idx   = sims.argmax().item()
                label = self._classes[idx]
            else:
                label = "unknown"

        return emb_np, label

    # ── Grid fallback (no MobileSAM) ─────────────────────────────────────────

    def _grid_fallback(self, rgb_img: np.ndarray) -> tuple:
        """
        Divide the image into a 4×4 grid of rectangular regions.
        Used when MobileSAM is not installed (development / CI).
        """
        H, W  = rgb_img.shape[:2]
        gh    = H // 4
        gw    = W // 4
        masks, scores, labels, embs = [], [], [], []

        for i in range(4):
            for j in range(4):
                mask = np.zeros((H, W), dtype=bool)
                mask[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = True
                emb, label = self._embed_and_classify(rgb_img, mask)
                masks.append(mask)
                scores.append(0.5)
                labels.append(label)
                embs.append(emb)

        return masks, scores, labels, embs
