#!/usr/bin/env python3
"""
Convert the ObChange dataset (real robot visits, oracle segmentation) into
SessionData files: <out>/<Room>/session_<kk>.npz, plus a report with the
ground-truth transitions and the CLIP zero-shot labels.

  python scripts/obchange_to_sessions.py \
      --root $HIPER_DATA/obchange/extracted \
      --out  $HIPER_DATA/obchange/sessions
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import conftest  # noqa: E402,F401  (registers the hiper package)

from hiper.evaluation import obchange as ob  # noqa: E402
from hiper.evaluation.object_session import save_session  # noqa: E402
from hiper.evaluation import paths as hpaths  # noqa: E402

TYPES_JSON = str(hpaths.THOR_TYPES_JSON)


def humanise(name: str) -> str:
    """'AlarmClock' -> 'alarm clock' (vocabulary entry for CLIP prompts)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
                  " ", name).lower()


class Embedder:
    """CLIP ViT-B/32 (OpenAI weights, QuickGELU) + DINOv2 ViT-B/14."""

    def __init__(self, device: str = "cuda"):
        import open_clip
        import timm
        import torch

        self.torch, self.device = torch, device
        self.clip, _, self.pre = open_clip.create_model_and_transforms(
            ob.CLIP_MODEL, pretrained=ob.CLIP_PRETRAINED, device=device)
        self.clip.eval()
        tok = open_clip.get_tokenizer(ob.CLIP_MODEL)
        self.vocab = sorted(json.load(open(TYPES_JSON))["types"])
        with torch.no_grad():
            feats = []
            for tmpl in ob.CLIP_PROMPTS:
                t = tok([tmpl.format(humanise(v)) for v in self.vocab])
                f = self.clip.encode_text(t.to(device)).float()
                feats.append(torch.nn.functional.normalize(f, dim=-1))
            text = torch.stack(feats).mean(0)
            self.text = torch.nn.functional.normalize(text, dim=-1)
        self.dino = timm.create_model(ob.DINO_MODEL, pretrained=True,
                                      num_classes=0, img_size=224
                                      ).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.dino)
        cfg["input_size"] = (3, 224, 224)
        self.dino_pre = timm.data.create_transform(**cfg)

    def __call__(self, images):
        from PIL import Image
        torch = self.torch
        pil = [Image.fromarray(im) for im in images]
        with torch.no_grad():
            x = torch.stack([self.pre(p) for p in pil]).to(self.device)
            e = torch.nn.functional.normalize(
                self.clip.encode_image(x).float(), dim=-1)
            probs = (ob.CLIP_LOGIT_SCALE * e @ self.text.T).softmax(-1)
            d = torch.stack([self.dino_pre(p) for p in pil]).to(self.device)
            g = torch.nn.functional.normalize(self.dino(d).float(), dim=-1)
        return {"clip": e.cpu().numpy(), "label_probs": probs.cpu().numpy(),
                "vocab": self.vocab, "dino": g.cpu().numpy()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(hpaths.OBCHANGE_EXTRACTED))
    ap.add_argument("--out", default=str(hpaths.OBCHANGE_SESSIONS))
    ap.add_argument("--rooms", nargs="*", default=list(ob.ROOMS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-dir", default=None,
                    help="save the first object's renders per room here")
    args = ap.parse_args()

    emb = Embedder(args.device)
    report = {"rooms": {}, "label_counts": Counter(), "examples": {}}
    t0 = time.time()
    for room in args.rooms:
        sessions, log = ob.build_room_sessions(args.root, room, emb)
        for s in sessions:
            save_session(s, os.path.join(args.out, room,
                                         f"session_{s.session_index:02d}.npz"))
            for tr in s.tracks:
                report["label_counts"][tr.clip_label] += 1
                report["examples"].setdefault(tr.gt_id, Counter())[
                    tr.clip_label] += 1
        trans = []
        for k in range(1, len(sessions)):
            g = {e["gt_id"]: e for e in sessions[k].gt_objects}
            gp = {e["gt_id"]: e for e in sessions[k - 1].gt_objects}
            trans.append({
                "k": k,
                "static": sorted(n for n, e in g.items() if e["present"]
                                 and gp.get(n, {}).get("present")
                                 and not e["moved_since_prev"]),
                "moved": sorted(n for n, e in g.items()
                                if e["moved_since_prev"]),
                "removed": sorted(n for n, e in g.items()
                                  if e["disappeared_since_prev"]),
                "novel": sorted(n for n, e in g.items()
                                if e["appeared_since_prev"])})
        report["rooms"][room] = {
            "visits": [s.meta["visit"] for s in sessions],
            "objects_per_visit": [len(s.tracks) for s in sessions],
            "transitions": trans, "log": log}
        if args.render_dir and sessions and sessions[0].tracks:
            from PIL import Image
            visit = ob.load_visit(ob.list_visits(args.root, room)[0], 0)
            name, o = sorted(visit.objects.items())[0]
            os.makedirs(args.render_dir, exist_ok=True)
            for i, im in enumerate(ob.object_views(o.xyz, o.rgb)[:4]):
                Image.fromarray(im).save(os.path.join(
                    args.render_dir, f"{room}_{name}_{i}.png"))
        print(room, report["rooms"][room]["objects_per_visit"], flush=True)
    report["label_counts"] = dict(report["label_counts"].most_common())
    report["examples"] = {k: dict(v) for k, v in report["examples"].items()}
    report["seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("done in", report["seconds"], "s")


if __name__ == "__main__":
    main()
