"""Input provenance, compact state recording, and structural metrics."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((array.shape, str(array.dtype))).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), package / "experiments/substrate_sweep.py",
             package / "model/inference_sampler.py", package / "engine.py",
             package / "system/chains.py", package / "system/proxy.py"]
    return {str(path.relative_to(package)): sha256(path) for path in files}


def read_cif(path):
    from biotite.structure.io.pdbx import CIFFile, get_structure
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return get_structure(CIFFile.read(handle), model=1, include_bonds=True)


def input_information(project, config):
    """Verify authoritative hashes/graphs and derive one shared origin."""
    from rdkit import Chem
    from biotite.structure import BondType, filter_amino_acids
    from rfd3_system_early_cut.inference.input_parsing import DesignInputSpecification

    root = Path(project) / config["input_dir"]
    source = read_json(root / "manifest.json")
    entries = {}
    for name in config["track_ligands"]:
        item = source["ligands"][name]
        for suffix in ("cif", "sdf"):
            require(sha256(root / f"{name}.{suffix}") == item[f"{suffix}_sha256"],
                    f"Input hash mismatch: {name}.{suffix}")
        spec = DesignInputSpecification.safe_init(input=str(root / f"{name}.cif"),
                                                  length=config["protein_length"], ligand=item["resname"])
        atoms = spec.atom_array_input
        require(not filter_amino_acids(atoms).any() and set(atoms.chain_id) == {"L"},
                "Ligand-only inputs required; no enzyme template may enter this sweep")
        require(set(atoms.atom_name) == set(item["atom_names"]), "Ligand atoms changed during parsing")
        require(atoms.bonds is not None, "Missing ligand bonds")
        rw = Chem.RWMol()
        for atom in atoms:
            element = Chem.Atom(str(atom.element))
            if "charge" in atoms.get_annotation_categories():
                element.SetFormalCharge(int(atom.charge))
            rw.AddAtom(element)
        kinds = {"SINGLE": Chem.BondType.SINGLE, "DOUBLE": Chem.BondType.DOUBLE,
                 "TRIPLE": Chem.BondType.TRIPLE}
        for a, b, kind in atoms.bonds.as_array():
            label = BondType(int(kind)).name
            rw.AddBond(int(a), int(b), Chem.BondType.AROMATIC if "AROMATIC" in label else kinds[label])
        molecule = rw.GetMol()
        Chem.SanitizeMol(molecule)
        expected = Chem.MolFromMolFile(str(root / f"{name}.sdf"))
        require(Chem.MolToSmiles(molecule) == Chem.MolToSmiles(expected), "Parsed ligand chemistry differs from SDF")
        entries[name] = dict(item, coordinates={str(a.atom_name): a.coord.tolist() for a in atoms})
    left, right = [entries[name]["coordinates"] for name in config["track_ligands"]]
    common = sorted(set(left) & set(right))
    require(len(common) == 16, "Expected 16 common substrate atoms")
    x, y = np.array([left[k] for k in common]), np.array([right[k] for k in common])
    require(np.array_equal(x, y), "Substrates are not in the same aligned frame")
    return dict(ligands=entries, common_atoms=common, origin=x.mean(axis=0).tolist())


class StateRecorder:
    """Collect actual CA states and exact shared-noise fingerprints, without RNG."""

    def __init__(self):
        self.ca = []
        self.sigmas = []
        self.noise_hashes = []
        self.initial_hash = None
        self.fixed_reference = None
        self.max_fixed_drift = 0.0

    def __call__(self, event):
        step = event["completed_updates"]
        require(step == len(self.ca), "Missing or duplicated state callback")
        require(event["coordinate_dtype"] == "torch.float32", "Sampler geometry must stay FP32")
        ca = np.stack([event[f"ca_{track}"].numpy()[0] for track in (1, 2)])
        require(np.isfinite(ca).all(), "Nonfinite protein coordinates")
        self.ca.append(ca.copy())
        self.sigmas.append(event["sigma"])
        fixed = [event[f"fixed_{track}"].numpy().copy() for track in (1, 2)]
        if step == 0:
            a, b = [event[f"initial_shared_{track}"].numpy() for track in (1, 2)]
            require(np.array_equal(a, b), "Protein initialization differs between tracks")
            self.initial_hash = digest(a)
            self.fixed_reference = fixed
        else:
            a, b = [event[f"noise_{track}"].numpy() for track in (1, 2)]
            require(np.array_equal(a, b), "Applied protein churn differs between tracks")
            self.noise_hashes.append(digest(a))
            drift = max((float(np.max(np.abs(a - b))) for a, b in zip(fixed, self.fixed_reference) if a.size), default=0.0)
            self.max_fixed_drift = max(self.max_fixed_drift, drift)
            require(drift < 1e-4, f"Fixed ligand moved by {drift} Angstrom")

    def audit(self):
        return dict(initial_shared_sha256=self.initial_hash, churn_sha256=self.noise_hashes,
                    maximum_fixed_coordinate_drift=self.max_fixed_drift, state_count=len(self.ca))

    def save(self, directory):
        np.savez_compressed(Path(directory) / "ca_states.npz", ca=np.stack(self.ca),
                            sigma=np.array(self.sigmas), completed_updates=np.arange(len(self.ca)))
        write_json(Path(directory) / "noise_audit.json", self.audit())


def rmsds(x, y):
    """Common-frame and proper-rotation Kabsch RMSD; arbitrary leading axes."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    raw = np.sqrt(np.mean(np.sum((x-y)**2, axis=-1), axis=-1))
    a, b = x - x.mean(axis=-2, keepdims=True), y - y.mean(axis=-2, keepdims=True)
    u, _, vh = np.linalg.svd(np.swapaxes(a, -1, -2) @ b)
    sign = np.linalg.det(u @ vh)
    u[..., :, -1] *= sign[..., None]
    aligned = np.sqrt(np.mean(np.sum((a @ (u @ vh) - b)**2, axis=-1), axis=-1))
    return raw, aligned


def jaccard(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    total = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / total) if total else None


def secondary_structure(ca):
    from biotite.structure import AtomArray, annotate_sse
    atoms = AtomArray(len(ca))
    atoms.coord = np.asarray(ca)
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    atoms.res_name[:] = "ALA"  # CA geometry only; native sequence does not enter this assignment.
    atoms.chain_id[:] = "A"
    atoms.res_id = np.arange(1, len(ca)+1)
    return np.asarray(annotate_sse(atoms))


def topology(x, y):
    a, b = secondary_structure(x), secondary_structure(y)
    valid = np.isin(a, ["a", "b", "c"]) & np.isin(b, ["a", "b", "c"])
    structured = valid & (np.isin(a, ["a", "b"]) | np.isin(b, ["a", "b"]))
    result = {"secondary_agreement": float(np.mean(a[valid] == b[valid])) if valid.any() else None,
              "secondary_valid_positions": int(valid.sum()),
              "structured_agreement": float(np.sum(structured & (a == b))/structured.sum()) if structured.any() else None,
              "helix_jaccard": jaccard((a == "a") & valid, (b == "a") & valid),
              "strand_jaccard": jaccard((a == "b") & valid, (b == "b") & valid)}
    contacts = []
    for label, coords, ss in (("ac", x, a), ("bu", y, b)):
        for kind, code in (("helix", "a"), ("strand", "b"), ("coil", "c")):
            result[f"{label}_{kind}_fraction"] = float(np.mean(ss == code))
        result[f"{label}_secondary_sequence"] = "".join(ss)
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        contact = (distances < 8.0) & (np.arange(len(coords))[None, :] - np.arange(len(coords))[:, None] >= 6)
        contacts.append(contact)
        result[f"{label}_long_range_contacts"] = int(contact.sum())
    result["contact_jaccard"] = jaccard(*contacts)
    return result
