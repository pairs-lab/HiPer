<p align="center">

  <h1 align="center">HiPer: Which Objects Stay?<br>Learning Hierarchical Persistence Beliefs for Lifelong Open-Vocabulary Object Maps</h1>
  <p align="center">
    <strong>Anonymous Authors</strong>
  </p>
  <p align="center"><strong>Submitted to IEEE Robotics and Automation Letters (RA-L), 2026</strong></p>
  <h3 align="center">Paper (coming soon) | <a href="./media/HiPer_video.mp4">Video</a> | <a href="#data">Data</a></h3>
  <div align="center"></div>
</p>
<p align="center">
    <img src="./media/teaser.png" alt="HiPer teaser" width="100%">
</p>

<p align="center">
HiPer predicts which mapped objects will still be where they were last seen.
</p>
<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quick-start">Quick start</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#reproduce-the-paper">Reproduce the paper</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#ros-integration-experimental">ROS integration (experimental)</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Overview

HiPer is a **hi**erarchical **per**sistence filter for lifelong object maps:

- **Prior retrieved by appearance and size.** A new object's survival rate is estimated from its K nearest training
  objects under a combined DINOv2 + CLIP + geometry (extents, height, mask area) distance: their weighted,
  interval-censored exponential maximum-likelihood rate with Gamma shrinkage to the pooled rate. No class label is
  needed, so zero-shot label noise does not enter the prior.
- **Object level.** That rate becomes a conjugate Beta prior over the object's own survival rate. Both the rate and
  the object's presence are updated in closed form from in-view detections and misses.
- **Relocation.** When an object is re-identified somewhere else, its learned rate moves with it; only its presence
  restarts.
- **Environment level (optional).** An online hazard multiplier, learned from all objects of a map, adapts the prior
  to a world that changes faster or slower than the training scenes.


## Installation

Tested on Ubuntu 20.04, Python 3.12, CUDA 12.8 and an RTX 4070 Ti.

1. Clone the repository.
```bash
git clone https://github.com/pairs-lab/HiPer.git
cd HiPer
```

2. Create a Python 3.12 environment (we use [uv](https://github.com/astral-sh/uv); conda works too).
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

3. Install PyTorch for CUDA 12.8, then the pinned evaluation dependencies.
```bash
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements-eval.txt
```
[requirements-eval.txt](requirements-eval.txt) pins the exact versions that produced the paper's numbers.
Two pins matter for correctness:
- **CLIP** must be built as `ViT-B-32-quickgelu` with the OpenAI weights. The plain `ViT-B-32` model uses the
  wrong activation.
- **numpy** must stay below 2.

4. Check the installation (285 tests, CPU only, about 8 s).
```bash
python -m pytest tests/ -q -p no:cacheprovider
```

The code is used straight from the source tree, without a pip install. Scripts register `src/` as the `hiper`
package through `tests/conftest.py`.

## Quick start

```python
import sys; sys.path.insert(0, "tests")
import conftest  # noqa: F401  (registers src/ as the `hiper` package)

import numpy as np
from hiper.ephemerality.persistence_v2 import HIT, MISS, HierarchicalPersistenceFilter
from hiper.ephemerality.priors import NeighbourSurvivalPrior

DAY = 86400.0
rng = np.random.default_rng(0)
# Training objects: appearance embedding + recorded history (survived exposure S,
# gaps that ended in a vanish). In the paper: priors.object_exposures(train sessions).
mugs = [{"emb": [1, 0, 0] + 0.05 * rng.standard_normal(3), "S": 2 * DAY, "gaps": [DAY]} for _ in range(20)]
sofas = [{"emb": [0, 1, 0] + 0.05 * rng.standard_normal(3), "S": 30 * DAY, "gaps": []} for _ in range(20)]
prior = NeighbourSurvivalPrior([{**r, "emb": r["emb"] / np.linalg.norm(r["emb"])} for r in mugs + sofas])

hpf = HierarchicalPersistenceFilter(
    rate_fn=lambda e: prior.rate(e, K=30, tau=0.05, k=0.25),   # retrieved prior
    default_lambda=prior.lam0, use_class_prior=False,
    kappa0=32, p_miss=0.2, p_false=0.001, tau_pi=0.5, variance_gate=False)

for oid, emb in (("mug_7", [0.98, 0.05, 0.0]), ("sofa_2", [0.02, 1.0, 0.03])):
    hpf.create(oid, "", t=0.0, emb=np.array(emb))   # embedding of the creating track
    hpf.observe(oid, 0.0, HIT)
    print(oid, f"P(still there tomorrow) = {hpf.presence(oid, DAY):.3f}")   # 0.705 / 1.000
hpf.observe("mug_7", DAY, MISS)                      # in view, not detected
print("mug_7 after a miss:", f"{hpf.presence('mug_7', 2 * DAY):.3f}")   # 0.224
```

The example uses DINOv2-like keys only. The paper's prior also uses CLIP and geometry keys:
`NeighbourSurvivalPrior(records, clip_weight=0.25, geo_weight=1.0)` with records from `priors.object_exposures`,
queried with `{"dino", "clip", "geo"}`. Parameters selected on the validation split: `kappa0=32`, `p_miss=0.2`,
`p_false=0.001`, `tau_pi=0.5`, `K=30`, `tau=0.05`, `k=0.25`, retrieval key DINOv2 + CLIP + geometry (the key is
tuned like any other parameter). Pass `env_alpha=5.0` to add the environment level. Optional post-hoc calibration
(a logistic map of the log-odds per last-evidence type, fitted on validation predictions) is in
[calibration.py](src/evaluation/calibration.py). Every
baseline in the paper implements the same `PersistenceModel` interface:
[persistence_v2.py](src/ephemerality/persistence_v2.py), [persistence_baselines.py](src/ephemerality/persistence_baselines.py),
[persistence_pocd.py](src/ephemerality/persistence_pocd.py); the prior is in [priors.py](src/ephemerality/priors.py).

## Data

All data goes under `$HIPER_DATA` (default `~/lifelong_ws/data`):
```
$HIPER_DATA/hiper_thor/{raw,sessions,results,object_types.json}
$HIPER_DATA/obchange/{extracted,sessions}
```

### THOR-Lifelong (simulation)

The benchmark is generated with AI2-THOR 5.0.0.
```bash
# raw RGB-D sessions + ground truth (~2.1 GB and ~1-2 h per 20 sequences on our shared machine)
for s in train train2 val test test2 test3 test4 test5; do python scripts/thor_generate_sessions.py --split $s --sessions 8 --jobs 2; done
# object tracks with CLIP + DINOv2 features -> sessions/<split>/<scene>/session_<kk>.npz
python scripts/thor_extract_tracks.py --splits train train2 val test test2 test3 test4 test5 --workers 6
```

### ObChange (real robot)

Download the [ObChange dataset](https://doi.org/10.48436/y3ggy-hxp10) (Langer et al., TU Wien, CC BY 4.0) and extract
it to `$HIPER_DATA/obchange/extracted/<Room>/<sceneN>/`. Then convert it:
```bash
python scripts/obchange_to_sessions.py   # oracle segmentation, rendered CLIP/DINOv2 views
```

### 3RScan (real)

Fill in the [3RScan Terms of Use](https://github.com/WaldJohannaU/3RScan) form. Put the public metadata file at
`$HIPER_DATA/3rscan/public/3RScan.json` (link in the 3RScan repository's `setup.sh`). Then download the files the adapter
needs (63 GB) and convert them. The protocol is documented in [rscan.py](src/evaluation/rscan.py).
```bash
python scripts/rscan_download.py --splits rscan_train rscan_val rscan_test   # sequence.zip unpacked, labels, semseg
python scripts/rscan_to_sessions.py --splits rscan_train rscan_val
python scripts/rscan_fit_cov.py --apply rscan_train rscan_val                 # partial-scan covariance, train only
```

## Reproduce the paper

```bash
python scripts/tune_and_eval.py tune       --run v5   # association + first persistence grids (val)
python scripts/tune_and_eval.py tune_final --run v5   # every persistence model on val -> final.json
python scripts/tune_and_eval.py calibrate  --run v5   # calibration maps on val -> calib.json
python scripts/tune_and_eval.py test       --run v5 --splits test3 test4 test5   # held-out, after freezing
python scripts/pool_heldout.py             --run v5 --splits test3 test4 test5   # -> results/v5/heldout
python scripts/tune_and_eval.py test       --run v5 --splits obchange            # real robot, tuned in simulation
python scripts/tune_and_eval.py loro       --run v6   # ObChange leave-one-room-out, cross-fitted priors
                                                      # (copy v5/final.json and v5/calib.json into results/v6 first)
python scripts/tune_and_eval.py test       --run v3 --splits test test2   # development splits (paper: run v3)
python scripts/tune_and_eval.py trainsize  --run v3 --train-splits train train2   # val: 4..60 training floor plans
# 3RScan: tune on rscan_val with priors from rscan_train, freeze, then convert and test rscan_test once
python scripts/tune_and_eval.py tune --bench rscan --run rscan_v1   # then tune_final, calibrate (same flags)
python scripts/rscan_to_sessions.py --splits rscan_test --kappa 0.075   # after freezing final.json / calib.json
python scripts/tune_and_eval.py test --bench rscan --run rscan_v1 --splits rscan_test
```
In the paper each held-out set was first evaluated by one frozen round (test3 by v3, test4 by v4, test5 by v5);
validation selected the same THOR parameters in all three rounds, so `test3` and `test4` re-run with the v5
configuration give the same raw predictions and only add the calibrated rows.

## Results

Lower is better except AUROC. &dagger; marks a method that is significantly worse than HiPer
(paired bootstrap over sequences, 95% CI; calibrated column: against calibrated HiPer).

### THOR-Lifelong: 120 held-out sequences (three sets of 40, each evaluated once) and the two development test splits

| Method | NLL &darr; | Brier &darr; | AUROC &uarr; | NLL &darr; calibrated | NLL &darr; dev test | NLL &darr; dev test-B |
|---|---|---|---|---|---|---|
| **HiPer (ours)** | **0.265** | **0.081** | **0.911** | **0.260** | **0.270** | **0.245** |
| PF(&lambda;), relocation-aware | 0.327&dagger; | 0.093&dagger; | 0.830&dagger; | 0.311&dagger; | 0.331&dagger; | 0.298&dagger; |
| PF(&lambda;) [Rosen et al. 2016] | 0.341&dagger; | 0.097&dagger; | 0.786&dagger; | 0.318&dagger; | 0.340&dagger; | 0.313&dagger; |
| ELite-style scalar | 0.365&dagger; | 0.104&dagger; | 0.773&dagger; | 0.341&dagger; | 0.363&dagger; | 0.340&dagger; |
| POCD-style | 0.402&dagger; | 0.127&dagger; | 0.784&dagger; | 0.326&dagger; | 0.397&dagger; | 0.387&dagger; |
| *Ablations of HiPer (re-tuned)* | | | | | | |
| &nbsp;&nbsp;PF, retrieved rates (no per-object learning) | 0.274&dagger; | 0.083&dagger; | 0.905&dagger; | 0.263&dagger; | 0.279&dagger; | 0.255&dagger; |
| &nbsp;&nbsp;HiPer, CLIP class rates (first version) | 0.296&dagger; | 0.085&dagger; | 0.887&dagger; | 0.284&dagger; | 0.296&dagger; | 0.263&dagger; |
| &nbsp;&nbsp;PF, CLIP class rates | 0.310&dagger; | 0.091&dagger; | 0.869&dagger; | 0.291&dagger; | 0.310&dagger; | 0.282&dagger; |

PF(&lambda;) has the lowest calibration error (ECE 0.045 vs 0.050 raw, 0.041 vs 0.050 calibrated) but ranks objects
far worse. Ablations on the held-out sequences (NLL): retrieval by DINOv2 only 0.270, rate reset at relocation
0.270, one pooled rate 0.326. A diagnostic filter given every object's TRUE simulated rate is not better in NLL
(validation 0.288 vs 0.274) but ranks slightly better (AUROC 0.913 vs 0.906).

### 3RScan, real indoor environments: 47 test environments, evaluated once

| Method | NLL &darr; | Brier &darr; | ECE &darr; | AUROC &uarr; | NLL &darr; calibrated |
|---|---|---|---|---|---|
| **HiPer (ours)** | **0.310** | **0.093** | 0.122 | **0.790** | 0.311 |
| PF(&lambda;), relocation-aware | 0.374&dagger; | 0.108&dagger; | 0.099 (better) | 0.555&dagger; | 0.371&dagger; |
| PF(&lambda;) [Rosen et al. 2016] | 0.377&dagger; | 0.109&dagger; | 0.098 (better) | 0.522&dagger; | 0.369&dagger; |
| ELite-style scalar | 0.382&dagger; | 0.110&dagger; | 0.117 | 0.551&dagger; | 0.371&dagger; |
| POCD-style | 0.375&dagger; | 0.108&dagger; | 0.102 (better) | 0.650&dagger; | 0.364&dagger; |


### ObChange real robot (5 rooms): the gains do not carry over

| Method | NLL &darr;, tuned in simulation | NLL &darr;, leave-one-room-out | AUROC &uarr;, leave-one-room-out | NLL &darr;, LORO calibrated |
|---|---|---|---|---|
| HiPer (ours) | 1.528 | 0.498 | 0.630 | 0.487 |
| PF(&lambda;) | 2.398 | **0.427** | 0.544 | 0.446 |
| PF(&lambda;), relocation-aware | 2.261 | 0.469 | 0.623 | **0.434** |
| ELite-style scalar | 1.258 | 0.545 | 0.619 | 0.446 |
| Never forget | 11.597 | 11.597 | 0.500 | 0.445 |


## ROS integration (experimental)

The repository is also a ROS Noetic catkin package. It contains an earlier online pipeline: VIO front end,
open-vocabulary segmentation, object re-ID and a long-term map manager. **This pipeline was not used for, or
evaluated in, the paper.**
```bash
cd ~/catkin_ws/src && ln -s /path/to/HiPer . && cd .. && catkin_make
roslaunch hiper hiper_desktop.launch session_id:=0        # Jetson: hiper_jetson.launch
```

## Repository structure

```
src/ephemerality/     HiPer filter (persistence_v2.py), baselines, class-rate learning (priors.py)
src/object_encoder/   cross-session association (association_v2.py)
src/evaluation/       THOR-Lifelong + ObChange adapters, mapping/replay, metrics, data paths (paths.py)
scripts/              data generation, conversion, tuning/evaluation
tests/                unit tests (pytest)
nodes/, launch/, msg/ ROS pipeline (experimental)
```

## Acknowledgement

We build on [AI2-THOR](https://ai2thor.allenai.org/) and the RoomR scene split
([Weihs et al., CVPR 2021](https://github.com/allenai/ai2thor-rearrangement)), the
[ObChange dataset](https://doi.org/10.48436/y3ggy-hxp10) (Langer et al.),
[OpenCLIP](https://github.com/mlfoundations/open_clip) and [DINOv2](https://github.com/facebookresearch/dinov2).
The persistence filter of Rosen et al. (ICRA 2016), and the POCD and ELite formulations, served as baselines.
Thanks to all authors for making their code and data public.

## Citation

If you find this work useful, please cite:
```bibtex
@misc{hiper2026,
  title  = {Which Objects Stay? Learning Hierarchical Persistence Beliefs for Lifelong Open-Vocabulary Object Maps},
  author = {Anonymous},
  note   = {Under review at IEEE Robotics and Automation Letters},
  year   = {2026}
}
```

## Contact

Please open a [GitHub issue](https://github.com/pairs-lab/HiPer/issues) for questions, comments and bug reports.
