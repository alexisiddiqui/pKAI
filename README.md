[![PyPI version](https://badge.fury.io/py/pKAI.svg)](https://badge.fury.io/py/pKAI) [![PyPI - Downloads](https://img.shields.io/pypi/dm/pKAI)](https://badge.fury.io/py/pKAI)

# pKAI

A fast and interpretable deep learning approach to accurate electrostatics-driven pKa prediction

```
@article{pkai,
author = {Reis, Pedro B. P. S. and Bertolini, Marco and Montanari, Floriane and Machuqueiro, Miguel and Clevert, Djork-Arné},
title = {pKAI: A fast and interpretable deep learning approach to accurate electrostatics-driven pKa prediction},
note = {in preparation}
}
```

### Installation & Basic Usage

We recommend installing pKAI on a conda enviroment. The pKAI+ model will be downloaded on the first execution and saved for subsequent runs.

```
python3 -m pip install pKAI

pKAI <pdbfile>
```

It can also be used as python function,
```
from pKAI.pKAI import pKAI

pks = pKAI(pdb)
```
where each element of the returned list is a tuple of size 4. (chain, resnumb, resname, pk)

## Retrained joint models in this fork

This fork adds two models trained from scratch with the pKAI architecture and a joint pKPDB/PINDER objective.
The original pKAI implementation and architecture are credited to Pedro B. P. S. Reis, Marco Bertolini,
Floriane Montanari, Miguel Machuqueiro, and Djork-Arné Clevert (Reis et al., *J. Chem. Theory Comput.*, 2022).
The original Bayer MIT licence and copyright notice are retained; it permits modification and redistribution
with those notices included.

| Model | Encoding / input width | pKPDB validation MSE | Model file size |
|---|---|---:|---:|
| `pKAI` (upstream reference) | atom16 / 4,008 | 0.324 (rounded) | unchanged |
| `pKAI+` (upstream reference) | atom16 / 4,008 | 0.802 (rounded) | unchanged |
| `pKAI-joint` | atom16 / 4,008 | 0.28650585953390484 | 14,450,346 bytes |
| `pKAI-joint-aa20` | atom16aa20 / 9,008 | 0.27910175357817624 | 30,450,346 bytes |

These numbers use the same fixed 7,778-site pKPDB validation set. They are distinct from the experimental PKAD
benchmark below: they do not establish improved accuracy on experimental pKa measurements. pKPDB labels are
PypKa-derived, and the PINDER part of the joint objective uses pKAI-derived labels.

Both checkpoints use 100% of the leakage-screened pool-v3 training subsets, the joint objective, all-atom
features, a 15 Å cutoff, 250 nearest-neighbour slots, batch size 4,096, learning rate `5.12e-4` with cosine decay
over 40 epochs, and seed 17. `pKAI-joint` uses the original input representation and is a drop-in model choice.
`pKAI-joint-aa20` adds 20 amino-acid channels after the 16 atom-class channels in each slot. Both active channels
receive `1/distance**2`; slots are sorted by distance, atom class, and amino-acid index, then truncated to 250.
The eight site-identity channels remain at the end. This extended encoder is used only for `pKAI-joint-aa20`.
The default model remains `pKAI`, and the original `pKAI` and `pKAI+` feature and model files are preserved.

To install this branch and select either new model:

```bash
python3 -m pip install 'git+https://github.com/alexisiddiqui/pKAI.git@joint-retrained-models'
pKAI protein.pdb --model pKAI-joint
pKAI protein.pdb --model pKAI-joint-aa20
```

```python
from pKAI.pKAI import pKAI
pks = pKAI("protein.pdb", model_name="pKAI-joint-aa20")
```

Training and export code lives in [jax-Ka at commit
`5ea7c176379635b3bb410cf9acd2c571af98616d`](https://github.com/alexisiddiqui/jax-Ka/tree/5ea7c176379635b3bb410cf9acd2c571af98616d),
in `src/pkatrain/pkai_scratch.py` and `src/pkatrain/pkai_joint_scale.py`. The selected checkpoints are
`training/pkai-joint-scale-v1-f100/runs/full-joint-b4096-e40-lr0.000512-cos/seed-17/best.pt` and
`training/pkai-joint-scale-v1-f100-atom16aa20/runs/full-joint-b4096-e40-lr0.000512-cos/seed-17/best.pt`
under the pkabench runtime.

The bundled TorchScript models accept float32 tensors with shape `(batch, input_width)` and return `(batch, 1)`
pKa shifts, matching the upstream forward API. Each export was reloaded and checked against a fresh `rescore`
result: both the maximum prediction difference and the MSE difference were exactly zero on all 7,778 validation
sites (required tolerance `1e-6`). Adjacent `models/*.export-check.json` files record exact metrics, checkpoint
and model SHA-256 hashes, sizes, and the export-code commit. The exports were produced with PyTorch 2.11.0;
use that version for the validated loading environment.

`tests/validate_release.py` checks three validation PDBs through both the Python API and CLI against an untouched
upstream checkout and the jax-Ka checkpoints. It also checks original model hashes, bit-identical original
features/predictions/output, aa20 aliases, equal-distance ordering, and 250-slot truncation. Run it in a Slurm
compute allocation with the validation inputs available; inference and training checks are not run on login nodes.
The committed [release validation receipt](tests/release-validation.json) records a passing check on three PDBs
(414 titratable sites): both new feature encodings and predictions matched jax-Ka exactly, and upstream pKAI and
pKAI+ outputs remained bit-identical.

## pKAI+ vs pKAI models

pKAI+ aims to predict experimental p<i>K</i><sub>a</sub> values from a single conformation. To do such, the interactions characterized in the input structure are given less weight and, as a consequence, the predictions are closer to the p<i>K</i><sub>a</sub> values of the residues in water. This effect is comparable to an increase in the dielectric constant of the protein in Poisson-Boltzmann models. In these models, the dielectric constant tries to capture, among others, electronic polarization and side-chain reorganization. When including conformational sampling explicitly, one should use a lower value for the dielectric constant of the protein. Likewise, one should use pKAI -- instead of pKAI+ -- as in this model there is no penalization of the interactions' impact on the predicted p<i>K</i><sub>a</sub> values.

tl;dr version
- use pKAI+ for p<i>K</i><sub>a</sub> predictions arising from a single structure
- use pKAI for p<i>K</i><sub>a</sub> predictions arising from multiple conformations

Change the model to be used in the calculation by evoking the `model` argument:
```
pKAI <pdbfile> --model pKAI
```

## Benchmark

Performed on 736 experimental values taken from the PKAD database<sup>1</sup>.

| Method                | RMSE | MAE  | Quantile 0.9  | Error < 0.5 (%)  |
|-----------------------|------|------|---------------|------------------|
| Null<sup>2</sup>      | 1.09 | 0.72 |          1.51 |             52.3 |
| PROPKA<sup>3</sup>    | 1.11 | 0.73 |          1.58 |             51.1 |
| PypKa<sup>4</sup>     | 1.07 | 0.71 |          1.48 |             52.6 |
| pKAI                  | 1.15 | 0.75 |          1.66 |             49.3 |
| pKAI+                 | 0.98 | 0.64 |          1.37 |             55.0 |

[1] Pahari, Swagata et al. "PKAD: a database of experimentally measured pKa values of ionizable groups in proteins." doi:<a href="https://doi.org/10.1093/database/baz024">10.1093/database/baz024</a>

[2] Thurlkill, Richard L et al. “pK values of the ionizable groups of proteins.” doi:<a href="https://doi.org/10.1110/ps.051840806">10.1110/ps.051840806</a>

[3] Olsson, Mats H M et al. “PROPKA3: Consistent Treatment of Internal and Surface Residues in Empirical pKa Predictions.” doi:<a href="https://doi.org/10.1021/ct100578z">10.1021/ct100578z</a>

[4] Reis, Pedro B P S et al. “PypKa: A Flexible Python Module for Poisson-Boltzmann-Based pKa Calculations.” doi:<a href="https://doi.org/10.1021/acs.jcim.0c00718">10.1021/acs.jcim.0c00718</a>


## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

## Contacts
Please submit a github issue to report bugs and to request new features. Alternatively, you may <a href="pdreis@fc.ul.pt"> email the developer directly</a>.
