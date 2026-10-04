"""Prepare immutable manifests; never regenerate an existing campaign."""
import argparse
from common import *
from rfd3_variants.conditions import conditions,ramp,calibration_key


def prepare(out,stage,calibration=None,selected=None):
    if stage=="transfer":
        raise NotImplementedError("SEP/SER transfer must use matched motif specifications after finalist selection; motif-free inputs are not a transfer experiment")
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing populated campaign directory {out}")
    if stage=="screen":
        if calibration is None:
            raise ValueError("screen requires a validated calibration file")
        gains=load(calibration)["gains"]
        configs=conditions()
        for row in configs:
            if "calibration_target" in row:
                row["experiment"]["strength"]=row["calibration_target"]/gains[calibration_key(row)]
        seeds=list(range(10001,10051))
    elif stage=="calibration":
        configs=[]
        for method,block in [("soft_guidance",1),("coarse_coupling",3),("coarse_coupling",5),("coarse_coupling",9),("sequence_coupling",1)]:
            row=dict(id=(f"coarse_{block}" if method=="coarse_coupling" else method),model="rfd3_"+method,
                experiment=dict(calibration=True,block=block,strength=0.))
            if method=="sequence_coupling":
                row["experiment"]["temperature"]=ramp(2.,1.)
            configs.append(row)
        seeds=[90001,90002,90003]
    elif stage=="smoke":
        configs=[conditions()[0]]
        seeds=[90011]
    elif stage in ("confirm","combinations","transfer"):
        if selected is None:
            raise ValueError("selected conditions file required")
        configs=load(selected)["conditions"]
        seeds=list(range(20001,20201)) if stage=="confirm" else list(range(30001,30201)) if stage=="combinations" else list(range(40001,40051))
    else:
        raise ValueError(stage)
    rows=[dict(index=i,condition=c,seed=s,replicate=k,stage=stage,debug=k<3)
          for i,(c,k,s) in enumerate((c,k,s) for c in configs for k,s in enumerate(seeds))]
    payload=dict(schema=1,stage=stage,source_revision=revision(),pipeline_sha256=controls(),conditions=configs,seeds=seeds,
        rows=rows,pairs=len(rows),sequence_candidates=4,expected_folds=0 if stage=="calibration" else len(rows)*8,
        sigma_values=200,chain_lengths=dict(A=90,B=80,C=100),threshold_angstrom=2.,
        folding_model="biohub/ESMFold2",calibration=str(calibration) if calibration else None)
    write(out/"manifest.json",payload,exclusive=True)
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--stage",choices=["smoke","calibration","screen","confirm","combinations","transfer"],required=True)
    p.add_argument("--calibration")
    p.add_argument("--selected")
    a=p.parse_args()
    result=prepare(a.out,a.stage,a.calibration,a.selected)
    print(json.dumps({k:result[k] for k in ("stage","pairs","expected_folds")}))
