"""Strict structural measurements; no missing-residue or failed-fold filtering."""
import gzip
import math
from pathlib import Path
import numpy as np


def structure(path):
    from Bio.PDB import MMCIFParser, PDBParser
    path=Path(path)
    parser=MMCIFParser(QUIET=True) if ".cif" in path.name else PDBParser(QUIET=True)
    with (gzip.open(path,"rt") if path.suffix==".gz" else path.open()) as handle:
        return next(parser.get_structure("structure",handle).get_models())


def coordinates(model,chain,atoms=("CA",),expected=None):
    residues=[r for r in model[chain] if "CA" in r]
    if expected is not None and len(residues)!=expected:
        raise ValueError(f"chain {chain}: {len(residues)} residues, expected {expected}")
    xyz=np.array([[r[a].coord for a in atoms] for r in residues],dtype=float).reshape(-1,3)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("missing or nonfinite coordinates")
    return xyz


def align(mobile,target):
    if mobile.shape!=target.shape or len(mobile)<3:
        raise ValueError("complete matched coordinate arrays required")
    x=mobile-mobile.mean(0); y=target-target.mean(0)
    u,_,vh=np.linalg.svd(x.T@y)
    flip=np.diag([1.,1.,np.linalg.det(u@vh)])
    return x@(u@flip@vh)+target.mean(0)


def rmsd(mobile,target):
    return float(np.sqrt(np.mean(np.sum((align(mobile,target)-target)**2,axis=1))))


def complex_rmsd(predicted,reference,mapping,lengths):
    # Concatenate before fitting: independently fitting chains would erase poses.
    p=np.concatenate([coordinates(predicted,a,expected=n) for (a,b),n in zip(mapping,lengths)])
    q=np.concatenate([coordinates(reference,b,expected=n) for (a,b),n in zip(mapping,lengths)])
    return rmsd(p,q)


def geometry(model,a,b):
    from scipy.spatial.distance import cdist
    def heavy(chain):
        return np.array([atom.coord for atom in model[chain].get_atoms() if atom.element not in ("H","D")],float)
    x,y=heavy(a),heavy(b)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite heavy atoms")
    d=cdist(x,y)
    ca=coordinates(model,a)
    return dict(interchain_heavy_pairs_lt_2A=int((d<2.).sum()),
        interchain_heavy_contacts_4p5A=int((d<4.5).sum()),
        A_ca_radius_gyration=float(np.sqrt(((ca-ca.mean(0))**2).sum(1).mean())))


def topology(model,chain):
    import biotite.structure as bs
    atoms=[a for r in model[chain] if "CA" in r for a in r.get_atoms() if a.name in ("N","CA","C","O")]
    arr=bs.AtomArray(len(atoms)); arr.coord=np.array([a.coord for a in atoms])
    arr.chain_id=np.array([chain]*len(atoms)); arr.res_id=np.array([a.parent.id[1] for a in atoms])
    arr.res_name=np.array([a.parent.resname for a in atoms]); arr.atom_name=np.array([a.name for a in atoms])
    arr.element=np.array([a.element for a in atoms]); arr.hetero=np.zeros(len(atoms),bool)
    return "".join(bs.annotate_sse(arr).tolist())


def wilson(passed,total):
    if not total:return [None,None]
    z=1.959963984540054; p=passed/total; d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-half,center+half]


def summarize_candidates(rows):
    """A shared sequence must pass both states; never mix different candidates."""
    return dict(AB_pass=any(r["AB_pass"] for r in rows),AC_pass=any(r["AC_pass"] for r in rows),
        dual_pass=any(r["AB_pass"] and r["AC_pass"] for r in rows),
        dual_candidates=sum(r["AB_pass"] and r["AC_pass"] for r in rows))
