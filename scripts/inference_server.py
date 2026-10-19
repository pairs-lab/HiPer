#!/usr/bin/env python3
"""
scripts/inference_server.py
============================
Python 3.12 inference server for HiPer.

Runs MobileSAM + CLIP in a separate Python 3.12 process and
serves results over a TCP socket to the ROS Noetic (Python 3.8)
segmentation_node.

This is the same socket-bridge architecture used in your SLIDE_SLAM
setup, extended with per-instance CLIP embeddings.

Protocol
--------
Client → Server:
  [4 bytes big-endian uint32 = payload_len]
  [payload_len bytes JSON]:
    {
      "h": int, "w": int, "enc": str,
      "data": base64-encoded image bytes,
      "conf": float, "max_inst": int
    }

Server → Client:
  [4 bytes big-endian uint32 = response_len]
  [response_len bytes JSON]:
    {
      "mask":             list[int] (H*W flattened uint16),
      "embeddings":       list[float] (N * embed_dim flattened),
      "instance_classes": dict {str(inst_id): str(class_label)},
      "instance_scores":  dict {str(inst_id): float},
      "num_instances":    int
    }

Usage
-----
  # On the Jetson (Python 3.12 environment):
  python3.12 scripts/inference_server.py --port 9876

  # Or with explicit model paths:
  python3.12 scripts/inference_server.py \
      --port 9876 \
      --sam-ckpt ~/.hiper/checkpoints/mobile_sam.pt \
      --clip-model ViT-B/32 \
      --device cuda
"""

import argparse
import base64
import json
import logging
import signal
import socket
import struct
import sys
import threading
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[inference_server] %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper (mirrors mobile_sam_wrapper.py for Python 3.12)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationEngine:
    """
    Holds MobileSAM + CLIP and exposes a process() method.
    All heavy imports are deferred to __init__ so the import errors
    are caught cleanly.
    """

    DEFAULT_CLASSES = [
        "tree", "building", "person", "car", "bicycle",
        "bench", "lamppost", "trash can", "table", "chair",
        "wall", "fence", "road", "grass", "sky",
        "door", "window", "sign", "bus", "motorcycle",
    ]

    def __init__(self, sam_ckpt: Optional[str], clip_model: str,
                 device: str, confidence: float, max_instances: int):
        self._device     = device
        self._conf       = confidence
        self._max_inst   = max_instances
        self._sam        = None
        self._clip       = None
        self._preprocess = None
        self._text_feats = None
        self._embed_dim  = 512
        self._classes    = self.DEFAULT_CLASSES

        self._load_sam(sam_ckpt)
        self._load_clip(clip_model)

    # ── Load MobileSAM ────────────────────────────────────────────────────────

    def _load_sam(self, ckpt: Optional[str]):
        try:
            from mobile_sam import (sam_model_registry,
                                    SamAutomaticMaskGenerator)
            import os
            import urllib.request

            if ckpt is None:
                ckpt = os.path.expanduser(
                    "~/.hiper/checkpoints/mobile_sam.pt")

            if not os.path.exists(ckpt):
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                log.info("Downloading MobileSAM weights...")
                urllib.request.urlretrieve(
                    "https://github.com/ChaoningZhang/MobileSAM/"
                    "raw/master/weights/mobile_sam.pt", ckpt)

            sam = sam_model_registry["vit_t"](checkpoint=ckpt)
            sam.to(self._device).eval()
            self._sam = SamAutomaticMaskGenerator(
                sam,
                points_per_side=16,
                pred_iou_thresh=self._conf,
                stability_score_thresh=0.85,
                min_mask_region_area=200,
            )
            log.info("MobileSAM loaded.")
        except Exception as e:
            log.warning(f"MobileSAM not available: {e}. Using grid fallback.")

    # ── Load CLIP ─────────────────────────────────────────────────────────────

    def _load_clip(self, model_name: str):
        import torch
        try:
            import clip as openai_clip
            model, prep = openai_clip.load(model_name, device=self._device)
            self._clip        = model
            self._preprocess  = prep
            self._embed_dim   = model.visual.output_dim
            self._clip_lib    = "openai"

            with torch.no_grad():
                tokens = openai_clip.tokenize(
                    [f"a photo of a {c}" for c in self._classes]
                ).to(self._device)
                tf = model.encode_text(tokens).float()
                tf /= tf.norm(dim=1, keepdim=True) + 1e-8
                self._text_feats = tf
            log.info(f"OpenAI CLIP {model_name} loaded, dim={self._embed_dim}.")

        except ImportError:
            try:
                import open_clip
                model, _, prep = open_clip.create_model_and_transforms(
                    "ViT-B-32-quickgelu", pretrained="openai",
                    device=self._device)  # OpenAI weights need QuickGELU
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self._clip        = model
                self._preprocess  = prep
                self._embed_dim   = 512
                self._clip_lib    = "openclip"

                with torch.no_grad():
                    tokens = tokenizer(
                        [f"a photo of a {c}" for c in self._classes]
                    ).to(self._device)
                    tf = model.encode_text(tokens).float()
                    tf /= tf.norm(dim=1, keepdim=True) + 1e-8
                    self._text_feats = tf
                log.info(f"open_clip ViT-B-32 loaded, dim={self._embed_dim}.")

            except Exception as e:
                log.warning(f"CLIP not available: {e}. Embeddings = zeros.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def process(self, rgb: np.ndarray,
                conf: float, max_inst: int) -> dict:
        """
        rgb : (H, W, 3) uint8
        Returns dict ready for JSON serialisation.
        """
        H, W = rgb.shape[:2]

        if self._sam is not None:
            results = self._sam.generate(rgb)
            results = sorted(results, key=lambda x: x["area"], reverse=True)
            results = results[:max_inst]
            masks   = [r["segmentation"] for r in results]
            scores  = [float(r["predicted_iou"]) for r in results]
        else:
            masks, scores = self._grid_fallback(H, W, max_inst)

        # Build combined mask image (instance id per pixel)
        mask_img = np.zeros((H, W), dtype=np.uint16)
        for i, m in enumerate(masks, start=1):
            mask_img[m] = i

        embeddings   = []
        inst_classes = {}
        inst_scores  = {}

        for i, (m, s) in enumerate(zip(masks, scores), start=1):
            emb, lbl = self._embed_region(rgb, m)
            embeddings.append(emb)
            inst_classes[str(i)] = lbl
            inst_scores[str(i)]  = s

        emb_flat = (np.stack(embeddings).flatten().tolist()
                    if embeddings else [])

        return {
            "mask":             mask_img.flatten().tolist(),
            "embeddings":       emb_flat,
            "instance_classes": inst_classes,
            "instance_scores":  inst_scores,
            "num_instances":    len(masks),
        }

    def _embed_region(self, rgb: np.ndarray,
                       mask: np.ndarray) -> tuple:
        """Embed the masked crop with CLIP and return (emb, label)."""
        import torch
        from PIL import Image as PILImage

        dim = self._embed_dim
        if self._clip is None:
            return np.zeros(dim, np.float32), "unknown"

        rows = np.any(mask, 1)
        cols = np.any(mask, 0)
        if not rows.any():
            return np.zeros(dim, np.float32), "unknown"

        r0, r1 = np.where(rows)[0][[0,-1]]
        c0, c1 = np.where(cols)[0][[0,-1]]
        pad = 8
        r0 = max(0, r0-pad)
        r1 = min(rgb.shape[0]-1, r1+pad)
        c0 = max(0, c0-pad)
        c1 = min(rgb.shape[1]-1, c1+pad)
        crop = rgb[r0:r1+1, c0:c1+1]
        if crop.size == 0:
            return np.zeros(dim, np.float32), "unknown"

        pil = PILImage.fromarray(crop)
        with torch.no_grad():
            inp = self._preprocess(pil).unsqueeze(0).to(self._device)
            feat = self._clip.encode_image(inp).float()
            feat /= feat.norm(dim=1, keepdim=True) + 1e-8
            emb = feat.squeeze(0).cpu().numpy().astype(np.float32)

            label = "unknown"
            if self._text_feats is not None:
                sims  = (feat @ self._text_feats.T).squeeze(0)
                label = self._classes[sims.argmax().item()]

        return emb, label

    @staticmethod
    def _grid_fallback(H: int, W: int, max_inst: int):
        gh, gw = H // 4, W // 4
        masks, scores = [], []
        for i in range(4):
            for j in range(4):
                if len(masks) >= max_inst:
                    break
                m = np.zeros((H, W), dtype=bool)
                m[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = True
                masks.append(m)
                scores.append(0.5)
        return masks, scores


# ─────────────────────────────────────────────────────────────────────────────
# TCP server
# ─────────────────────────────────────────────────────────────────────────────

def recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError("Client disconnected.")
        buf += chunk
    return buf


def handle_client(conn: socket.socket, addr,
                   engine: SegmentationEngine):
    try:
        # Read request
        raw_len = recv_exactly(conn, 4)
        length  = struct.unpack(">I", raw_len)[0]
        payload = recv_exactly(conn, length)
        req     = json.loads(payload.decode("utf-8"))

        H, W   = req["h"], req["w"]
        enc    = req["enc"]
        data   = base64.b64decode(req["data"])
        conf   = float(req.get("conf",     0.5))
        maxn   = int(req.get("max_inst", 32))

        # Decode image
        img_np = np.frombuffer(data, dtype=np.uint8)

        if enc in ("rgb8", "bgr8"):
            rgb = img_np.reshape(H, W, 3)
            if enc == "bgr8":
                rgb = rgb[:, :, ::-1].copy()
        elif enc == "mono8":
            mono = img_np.reshape(H, W)
            rgb  = np.stack([mono]*3, axis=-1)
        else:
            # Generic: try OpenCV decode
            import cv2
            rgb = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Run inference
        result = engine.process(rgb.astype(np.uint8), conf, maxn)

        # Send response
        resp    = json.dumps(result).encode("utf-8")
        conn.sendall(struct.pack(">I", len(resp)) + resp)

    except Exception as e:
        log.warning(f"Client {addr} error: {e}")
        # Send empty response so the client does not hang
        empty = json.dumps({
            "mask": [], "embeddings": [],
            "instance_classes": {}, "instance_scores": {},
            "num_instances": 0,
        }).encode("utf-8")
        try:
            conn.sendall(struct.pack(">I", len(empty)) + empty)
        except Exception:
            pass
    finally:
        conn.close()


def run_server(host: str, port: int, engine: SegmentationEngine):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    log.info(f"Listening on {host}:{port}")

    def _shutdown(sig, _):
        log.info("Shutting down.")
        srv.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        try:
            conn, addr = srv.accept()
            log.debug(f"Connection from {addr}")
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, engine),
                daemon=True)
            t.start()
        except OSError:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HiPer Python 3.12 inference server")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=9876)
    parser.add_argument("--sam-ckpt",   default=None,
                        help="Path to mobile_sam.pt checkpoint")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-inst",   type=int,   default=32)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info(f"Initialising segmentation engine "
             f"(device={args.device}, clip={args.clip_model})")
    engine = SegmentationEngine(
        sam_ckpt     = args.sam_ckpt,
        clip_model   = args.clip_model,
        device       = args.device,
        confidence   = args.confidence,
        max_instances= args.max_inst,
    )
    log.info("Engine ready.")
    run_server(args.host, args.port, engine)


if __name__ == "__main__":
    main()
