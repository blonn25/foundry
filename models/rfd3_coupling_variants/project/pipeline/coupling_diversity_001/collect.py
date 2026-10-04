"""Collect complete paired candidates, calibration coefficients, and raw diagnostics."""
import argparse
import csv
from collections import defaultdict
import numpy as np
from common import *
from metrics import *


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w") as handle:
        w=csv.DictWriter(handle,fieldnames=keys);w.writeheader();w.writerows(rows)


def coupling(parent):
    paths=list((parent/"rfd").glob("*_track1_model_0.json"))
    if len(paths)!=1:raise ValueError(f"expected one track metadata file: {paths}")
    payload=load(paths[0])
    return payload["coupling"] if "coupling" in payload else payload["metadata"]["coupling"]


def calibrate(run):
    manifest=load(run/"manifest.json");by_key=defaultdict(list);samples=[];noise_banks={}
    for row in manifest["rows"]:
        parent=case_dir(run,row);require_complete(parent/"rfd")
        metadata=coupling(parent);trace=metadata["trace"]
        if len(set(metadata["initial_A_hashes"]))!=1 or any(len(set(t["churn_A_hashes"]))!=1 for t in trace[1:]):
            raise ValueError("shared initialization/churn mismatch within calibration pair")
        bank=(metadata["initial_A_hashes"][0],[t["churn_A_hashes"][0] for t in trace[1:]])
        if row["seed"] in noise_banks and noise_banks[row["seed"]]!=bank:
            raise ValueError("calibration methods did not consume the same random noise")
        noise_banks[row["seed"]]=bank
        values=[max(t["unit_correction_ratios"]) for t in trace[1:] if .1<=t["progress"]<=.8]
        value=float(np.median(values))
        if not np.isfinite(value) or value<=0:raise ValueError(f"unusable guidance derivative: {row}")
        by_key[row["condition"]["id"]].append(value)
        samples.append(dict(condition=row["condition"]["id"],seed=row["seed"],median_unit_ratio=value))
    if any(len(v)!=3 for v in by_key.values()):raise ValueError("three calibration seeds required")
    gains={k:float(np.median(v)) for k,v in by_key.items()}
    write(run/"calibration.json",dict(gains=gains,samples=samples,source_revision=manifest["source_revision"],
        shared_noise_audit="identical within pairs and across methods for every matched seed",
        definition="median across three seeds of median max-state unit correction/native-update ratio over p=0.1..0.8"))
    write(run/"_COMPLETE.json",dict(calibrated=True))


def collect(run,allow_incomplete=False):
    manifest=load(run/"manifest.json");parents=[];candidates=[];errors=[]
    summary=run/"analysis";summary.mkdir(exist_ok=True)
    for row in manifest["rows"]:
        parent=case_dir(run,row);base=dict(condition=row["condition"]["id"],seed=row["seed"])
        try:
            raw=require_complete(parent/"rfd");folds=require_complete(parent/"folds")["folds"]
            if len(folds)!=8:raise ValueError("eight fold results required")
            refs=[structure(host(raw[f"track{i}"])) for i in (1,2)]
            groups=defaultdict(dict)
            for f in folds:groups[f["design_index"]][f["complex_kind"]]=f
            if len(groups)!=4 or any(set(g)!={"AB","DC"} for g in groups.values()):
                raise ValueError("four complete same-sequence pairs required")
            local=[]
            for index,states in sorted(groups.items()):
                item=base|dict(candidate=index)
                if states["AB"]["sequences"]["A"]!=states["DC"]["sequences"]["D"]:
                    raise ValueError("folded candidates do not share A sequence")
                for state,internal,ref,mapping,lengths in [
                    ("AB","AB",refs[0],[("A","A"),("B","B")],[90,80]),
                    ("AC","DC",refs[1],[("D","A"),("C","C")],[90,100])]:
                    f=states[internal]
                    if len(f["cifs"])!=1:raise ValueError("exactly one prediction per candidate/state required")
                    pred=structure(host(f["cifs"][0]))
                    value=complex_rmsd(pred,ref,mapping,lengths)
                    item[state+"_rmsd"]=value;item[state+"_pass"]=value<manifest["threshold_angstrom"]
                    for key,value in f["samples"][0].items():
                        if isinstance(value,(int,float)) and np.isfinite(value):item[state+"_"+key]=value
                    item[state+"_cif"]=f["cifs"][0]
                item["dual_pass"]=item["AB_pass"] and item["AC_pass"]
                local.append(item)
            a=[coordinates(r,"A",expected=90) for r in refs]
            bb=[coordinates(r,"A",("N","CA","C","O"),90) for r in refs]
            topo=[topology(r,"A") for r in refs]
            if any(len(t)!=90 for t in topo):raise ValueError("incomplete secondary-structure assignment")
            meta=coupling(parent)
            data=base|summarize_candidates(local)|dict(A_ca_rmsd=rmsd(*a),A_backbone_rmsd=rmsd(*bb),
                secondary_structure_agreement=sum(x==y for x,y in zip(*topo))/90,
                AB_topology=topo[0],AC_topology=topo[1],AB_reference=raw["track1"],AC_reference=raw["track2"],
                rfd_seconds=meta["elapsed_seconds"],initial_A_hashes=meta["initial_A_hashes"],
                best_joint_rmsd=min(max(r["AB_rmsd"],r["AC_rmsd"]) for r in local))
            for state,r,partner in [("AB",refs[0],"B"),("AC",refs[1],"C")]:
                data.update({state+"_"+k:v for k,v in geometry(r,"A",partner).items()})
                displacement=np.linalg.norm(align(a[0],a[1])-a[1],axis=1)
                write(parent/"A_displacement.json",dict(angstrom=displacement.tolist()))
            for stage in ("generate","prepare_caliby","caliby","fold"):
                data[stage+"_seconds"]=load(parent/(stage+"_job.json"))["seconds"]
            parents.append(data);candidates.extend(local)
        except Exception as exc:
            errors.append(base|dict(error=repr(exc)))
    write(summary/"errors.json",errors);write(summary/"parents.json",parents);write(summary/"candidates.json",candidates)
    csv_write(summary/"parents.csv",parents);csv_write(summary/"candidates.csv",candidates)
    by_condition=[]
    reference={p["seed"]:p for p in parents if p["condition"]=="mean_5050"}
    for config in manifest["conditions"]:
        ps=[p for p in parents if p["condition"]==config["id"]]
        cs=[c for c in candidates if c["condition"]==config["id"]]
        attempted=sum(r["condition"]["id"]==config["id"] for r in manifest["rows"])
        s=dict(condition=config["id"],model=config["model"],hyperparameters=config,attempted=attempted,complete=len(ps))
        for state in ("AB","AC","dual"):
            successes=sum(p[state+"_pass"] for p in ps)
            s[state+"_success_count"]=successes
            s[state+"_success_rate"]=successes/len(ps) if ps else None
            s[state+"_success_ci"]=wilson(successes,len(ps))
            s[state+"_candidate_success_rate"]=sum(c[state+"_pass"] for c in cs)/len(cs) if cs else None
        for key in ("A_ca_rmsd","A_backbone_rmsd","secondary_structure_agreement","rfd_seconds","fold_seconds"):
            s[key+"_mean"]=float(np.mean([p[key] for p in ps])) if ps else None
            s[key+"_median"]=float(np.median([p[key] for p in ps])) if ps else None
        matched=[p for p in ps if p["seed"] in reference]
        if matched:
            differences=np.array([int(p["dual_pass"])-int(reference[p["seed"]]["dual_pass"]) for p in matched])
            rng=np.random.default_rng(seed("paired_bootstrap",config["id"]))
            bootstrap=rng.choice(differences,(2000,len(differences)),replace=True).mean(1)
            s["paired_dual_rate_difference"]=float(differences.mean())
            s["paired_dual_rate_difference_ci"]=np.quantile(bootstrap,[.025,.975]).tolist()
        by_condition.append(s)
    write(summary/"conditions.json",by_condition);csv_write(summary/"conditions.csv",by_condition)
    if errors and not allow_incomplete:raise RuntimeError(f"{len(errors)} incomplete/invalid pairs; see errors.json")
    if not errors:write(run/"_COMPLETE.json",dict(pairs=len(parents),folds=len(candidates)*2))
    return parents,by_condition


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("--calibrate",action="store_true")
    p.add_argument("--allow-incomplete",action="store_true");a=p.parse_args()
    if a.calibrate:calibrate(a.run)
    else:collect(a.run,a.allow_incomplete)
