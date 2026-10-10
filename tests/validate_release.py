"""Validate a release against an untouched upstream checkout and jax-Ka checkpoints.

Run inside a Slurm allocation on Isambard with the validation dataset images mounted:
  python tests/validate_release.py --upstream UPSTREAM --training-repo JAXKA --runtime PKABENCH --output CHECK.json
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys


MODELS = {"pKAI-joint": "atom16", "pKAI-joint-aa20": "atom16aa20",
          "pKAI-joint-aa20-backbone": "atom16aa20"}
ARM = "full-joint-b4096-e40-lr0.000512-cos/seed-17"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_backbone_features(pdb, protein, encoding):
    """Use the actual training encoder with the PDB's decimal coordinates preserved as float64."""
    import numpy as np
    from types import SimpleNamespace
    from pkabench.pkai_backbone_pinder_eval import _features
    records = []
    with open(pdb) as handle:
        for line in handle:
            if line.startswith("ATOM "):
                record = protein.read_pdb_line(line)
                if record[2] in (" ", "A") and record[-1] == " ": records.append(record)
    atoms = SimpleNamespace(hetero=np.zeros(len(records), bool),
                            atom_name=np.asarray([r[0] for r in records]),
                            res_name=np.asarray([r[3] for r in records]),
                            chain_id=np.asarray([r[4] for r in records]),
                            res_id=np.asarray([r[5] for r in records]),
                            ins_code=np.full(len(records), ""),
                            coord=np.asarray([r[6:9] for r in records], dtype=np.float64))
    keys = [(r.chain, r.resnumb, "", r.resname) for r in protein.iter_residues(titrable_only=True)]
    features, retained = _features(atoms, keys, encoding)
    if not retained.all(): raise AssertionError("training encoder rejected backbone sites")
    return features


def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    package = Path(args.package)
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location("release_api", package / "pKAI.py")
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    from protein import Protein
    report = {"models": {}}
    for name in ("pKAI", "pKAI+", *MODELS):
        if name in MODELS and args.original: continue
        encoding = MODELS.get(name, "atom16")
        mode = "backbone" if name.endswith("-backbone") else "full"
        protein = Protein(args.pdb)
        if mode == "backbone": protein.apply_backbone_cutoff()
        elif encoding == "atom16": protein.apply_cutoff()
        else: protein.apply_cutoff(encoding=encoding)
        x = torch.stack([residue.input_layer for residue in protein.iter_residues(titrable_only=True)])
        model = api.load_model(name, "cpu")
        with torch.inference_mode(): prediction = model(x).reshape(-1)
        stdout = io.StringIO()
        with redirect_stdout(stdout): result = api.pKAI(args.pdb, model_name=name, threads=2)
        cli = subprocess.run([sys.executable, str(package / "pKAI.py"), args.pdb, "--model", name],
                             check=True, capture_output=True, text=True)
        if cli.stdout != stdout.getvalue(): raise AssertionError((name, "CLI/API output differs"))
        check = {"sites": len(x), "width": x.shape[1], "stdout": stdout.getvalue(), "cli_stdout": cli.stdout,
                 "result": result, "prediction_bits": prediction.numpy().tobytes().hex(),
                 "feature_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest()}
        if name in MODELS:
            sys.path.insert(0, str(Path(args.training_repo) / "src"))
            from pkatrain.pkai_scratch import feature_matrix, feature_width, model_class
            if mode == "backbone": expected_x = training_backbone_features(args.pdb, Protein(args.pdb), encoding)
            else: _, expected_x = feature_matrix(Protein(args.pdb), encoding)
            feature_error = float(np.max(np.abs(x.numpy() - expected_x)))
            np.testing.assert_allclose(x.numpy(), expected_x, rtol=0, atol=1e-7)
            suffix = "" if encoding == "atom16" else "-atom16aa20"
            arm = ARM.replace("full-", "backbone-", 1) if mode == "backbone" else ARM
            run = Path(args.runtime) / f"training/pkai-joint-scale-v1-f100{suffix}/runs" / arm
            eager = model_class(torch, inputs=feature_width(encoding))()
            eager.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)); eager.eval()
            with torch.inference_mode(): expected = eager(torch.from_numpy(expected_x))
            prediction_error = float((prediction - expected).abs().max())
            torch.testing.assert_close(prediction, expected, rtol=0, atol=1e-6)
            if len(result) != len(expected): raise AssertionError("missing predicted sites")
            from protein import PK_MODS
            expected_pkas = [round(float(value) + PK_MODS[residue.resname], 2)
                             for value, residue in zip(expected, protein.iter_residues(titrable_only=True))]
            if [row[3] for row in result] != expected_pkas: raise AssertionError((name, "rounded pKa differs"))
            check.update(feature_max_absolute_error=feature_error, prediction_max_absolute_error=prediction_error)
            if mode == "backbone":
                import tempfile
                lines = Path(args.pdb).read_text().splitlines(keepends=True)
                with tempfile.TemporaryDirectory() as directory:
                    stripped = Path(directory) / "backbone.pdb"
                    stripped.write_text("".join(line for line in lines if not line.startswith("ATOM ")
                                                or line[12:16].strip() in ("N", "CA", "C", "O")))
                    no_sidechain = Protein(stripped); no_sidechain.apply_backbone_cutoff()
                    stripped_x = torch.stack([r.input_layer for r in no_sidechain.iter_residues(titrable_only=True)])
                    assert torch.equal(x, stripped_x), "side chains changed backbone features"
                    with redirect_stdout(io.StringIO()): stripped_result = api.pKAI(stripped, model_name=name, threads=2)
                    assert stripped_result == result, "side chains changed backbone pKa output"
                    missing_ca = Path(directory) / "missing-ca.pdb"
                    missing_ca.write_text("".join(line for line in lines if not line.startswith("ATOM ")
                                                 or line[12:16].strip() != "CA"))
                    try: Protein(missing_ca).apply_backbone_cutoff()
                    except ValueError as error: assert "missing C-alpha" in str(error)
                    else: raise AssertionError("missing query CA was accepted")
                check.update(backbone_sidechains_ignored=True, missing_ca_rejected=True)
        report["models"][name] = check
    if not args.original:
        from residue import AA20_ALIASES, AA20, ATOM_OHE, RES_OHE, Residue, aa20_index
        for alias, canonical in AA20_ALIASES.items():
            assert aa20_index(alias) == AA20.index(canonical)
        try: aa20_index("UNK")
        except ValueError: pass
        else: raise AssertionError("unknown amino acid accepted")
        # Equal-distance/equal-class slots must use the same amino-acid tie break as training, then truncate at 250.
        residue = Residue(None, "A", "ASP", 1)
        residue.env_anames = ["OD1"] * 251
        residue.env_resnames = ["GLU", "ASP"] * 125 + ["ASP"]
        residue.env_dists = [2.0] * 251
        residue.encode_input_aa20()
        slots = residue.input_layer[:9000].reshape(250, 36)
        assert torch.all(slots[:, ATOM_OHE.index("O_COOH")] == 0.25)
        assert torch.all(slots[:126, 16 + AA20.index("ASP")] == 0.25)
        assert torch.all(slots[126:, 16 + AA20.index("GLU")] == 0.25)
        assert residue.input_layer[9000 + RES_OHE.index("ASP")] == 1
        report["aa20_aliases_and_tie_truncation"] = True
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream")
    parser.add_argument("--training-repo", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--package")
    parser.add_argument("--pdb")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("release validation requires a Slurm compute allocation")
    if args.worker:
        worker(args); return
    fork = Path(__file__).resolve().parents[1]
    upstream = Path(args.upstream)
    for name in ("pKAI", "pKAI+"):
        assert sha256(fork / f"pKAI/models/{name}_model.pt") == sha256(upstream / f"pKAI/models/{name}_model.pt")
    pilot = Path(args.runtime) / "pretraining/pkpdb-5k-comparison-v1"
    records = json.loads((pilot / "pkai-features.json").read_text())["records"]
    records = sorted((r for r in records if r["split"] == "val"), key=lambda r: r["complex_id"])[:3]
    if len(records) != 3: raise AssertionError("expected three validation PDBs")
    report = {"passed": True, "upstream_models_sha256_unchanged": True,
              "upstream_outputs_bit_identical": True, "prediction_tolerance": 1e-6, "structures": []}
    for record in records:
        pdb = pilot / "pkai-data/val" / Path(record["path"]).name / "input.pdb"
        checks = []
        for original, repo in ((True, upstream), (False, fork)):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--package", str(repo / "pKAI"),
                       "--pdb", str(pdb), "--training-repo", args.training_repo, "--runtime", args.runtime]
            if original: command.append("--original")
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode: raise RuntimeError(result.stderr)
            checks.append(json.loads(result.stdout))
        before, after = checks
        for name in ("pKAI", "pKAI+"):
            if before["models"][name] != after["models"][name]:
                raise AssertionError((record["complex_id"], name, "upstream output/feature bits changed"))
        report["structures"].append({"complex_id": record["complex_id"], "pdb_sha256": sha256(pdb),
                                     "models": {name: {key: value for key, value in check.items()
                                                       if key not in ("stdout", "cli_stdout", "prediction_bits", "result")}
                                                for name, check in after["models"].items()},
                                     "aa20_aliases_and_tie_truncation": after["aa20_aliases_and_tie_truncation"]})
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
