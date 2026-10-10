from residue import Residue
import torch
import math

PK_MODS = {
    "ASP": 3.79,
    "CTR": 2.90,
    "CYS": 8.67,
    "GLU": 4.20,
    "HIS": 6.74,
    "LYS": 10.46,
    "NTR": 7.99,
    "TYR": 9.59,
}


class Protein:
    def __init__(self, pdb_f):

        self.pdb_f = pdb_f
        self.tit_residues = {}
        self.all_residues = {}
        self.termini = {}  # TODO implement this later. ignored for now

        self.read_pdb()

    def iter_atoms(self):
        for residue in self.iter_residues():
            for atom in residue.iter_atoms():
                yield atom

    def iter_residues(self, titrable_only=False):
        if titrable_only:
            residues = self.tit_residues
        else:
            residues = self.all_residues
        for chain in sorted(residues.keys()):
            chain_residues = residues[chain]
            for resnumb in sorted(chain_residues.keys()):
                residue = chain_residues[resnumb]
                yield residue

    def read_pdb(self):
        """Removes all atoms from pdb_f that are not Nitrogens, Sulfurs, or Oxygens"""
        tit_aas = PK_MODS.keys()
        with open(self.pdb_f) as f:
            for line in f:
                if line.startswith("ATOM "):
                    line_cols = self.read_pdb_line(line)
                    (
                        aname,
                        anumb,
                        b,
                        resname,
                        chain,
                        resnumb,
                        x,
                        y,
                        z,
                        icode,
                    ) = line_cols

                    if b not in (" ", "A") or icode != " ":
                        continue

                    if aname[0] not in "NOS":
                        continue

                    if chain not in self.all_residues.keys():
                        self.all_residues[chain] = {}

                    if resnumb not in self.all_residues[chain].keys():
                        new_res = Residue(self, chain, resname, resnumb)
                        self.all_residues[chain][resnumb] = new_res
                    else:
                        new_res = self.all_residues[chain][resnumb]

                    new_res.add_atom(aname, anumb, x, y, z)

                    if resname in tit_aas:
                        if chain not in self.tit_residues.keys():
                            self.tit_residues[chain] = {}

                        if resnumb not in self.tit_residues[chain]:
                            self.tit_residues[chain][resnumb] = new_res

    @staticmethod
    def read_pdb_line(line: str) -> tuple:
        aname = line[12:16].strip()
        anumb = int(line[5:11].strip())
        b = line[16]
        resname = line[17:21].strip()
        chain = line[21]
        resnumb = int(line[22:26])
        icode = line[26]
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        return (aname, anumb, b, resname, chain, resnumb, x, y, z, icode)

    def apply_cutoff(self, cutoff_dist=15, encoding="atom16"):

        if encoding not in ("atom16", "atom16aa20"):
            raise ValueError(f"unsupported pKAI encoding: {encoding}")

        # TODO speed up by doing distance matrix a priori in cython

        for residue in self.iter_residues(titrable_only=True):
            residue.calc_cutoff_atoms(cutoff_dist)
            if encoding == "atom16aa20":
                residue.encode_input_aa20()
            else:
                residue.encode_input()

    def apply_backbone_cutoff(self, cutoff_dist=15):
        """Strict N/O backbone context centred on the query C-alpha, with atom16+aa20 slots."""
        ca = {}
        with open(self.pdb_f) as handle:
            for line in handle:
                if not line.startswith("ATOM ") or line[12:16].strip() != "CA":
                    continue
                _, _, alternate, _, chain, number, x, y, z, insertion = self.read_pdb_line(line)
                if alternate in (" ", "A") and insertion == " ":
                    ca[(chain, number)] = (x, y, z)
        context = [atom for atom in self.iter_atoms() if atom.aname in ("N", "O")]
        for residue in self.iter_residues(titrable_only=True):
            key = (residue.chain, residue.resnumb)
            if key not in ca:
                raise ValueError(f"missing C-alpha for backbone pKAI site {key}")
            origin = ca[key]
            for atom in context:
                if atom.residue is residue:
                    continue
                distance = math.sqrt(sum((value - center) ** 2 for value, center in zip(atom.coords, origin)))
                if distance < cutoff_dist:
                    residue.env_anames.append(atom.aname)
                    residue.env_resnames.append(atom.residue.resname)
                    residue.env_dists.append(distance)
            residue.encode_input_aa20()

    def predict_pkas(self, model, device):
        pks = []
        residues = []
        xs = []
        for residue in self.iter_residues(titrable_only=True):
            x = residue.input_layer
            xs.append(x)
            residues.append(residue)

        xs = torch.stack((xs), dim=0)
        dpks = model(xs.to(device))

        for residue, dpk in zip(residues, dpks):
            resname = residue.resname

            dpk = float(dpk)
            pk = round(dpk + PK_MODS[resname], 2)

            to_print = (
                f"{residue.chain:4} {residue.resnumb:6} {residue.resname:4} {pk:5.2f}"
            )
            print(to_print)

            result = (residue.chain, residue.resnumb, residue.resname, pk)
            pks.append(result)
        return pks
