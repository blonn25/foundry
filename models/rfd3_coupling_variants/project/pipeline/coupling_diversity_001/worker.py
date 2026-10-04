"""One independently checkpointed paired seed per SLURM array task."""
from __future__ import annotations
import argparse
import csv
import time
from common import *


def generate(run,row):
    c=row["condition"]
    out=case_dir(run,row)/"rfd"
    if (out/"_COMPLETE.json").exists():
        require_complete(out)
        return
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing scientific overwrite: {out}")
    cfg=dict(c.get("experiment",{}))
    args=["-m",c["model"]+".cli","design","inputs=null",f"out_dir={container(out)}",
        "global_prefix=pair","ckpt_path=/weights/rfd3_latest.ckpt",
        "+specification.contig=90,/0,80,/0,100","+specification.length=270",
        f"seed={row['seed']}",f"dump_trajectories={str(row['debug']).lower()}",
        "inference_sampler.num_timesteps=200","inference_sampler.step_scale=3",
        "inference_sampler.gamma_0=0.2","inference_sampler.gamma_min=1",
        "inference_sampler.noise_scale=1.003","inference_sampler.p=7",
        "inference_sampler.n_recycle=2","inference_sampler.use_classifier_free_guidance=false",
        "inference_sampler.allow_realignment=false","inference_sampler.s_jitter_origin=0",
        "shared_chain_id=A","complex_1_partners=[B]","complex_2_partners=[C]",
        "+track_1_specification.contig=A1-90,/0,B1-80","+track_1_specification.length=170",
        "+track_2_specification.contig=A1-90,/0,C1-100","+track_2_specification.length=190"]
    # Quotes belong to Hydra's grammar, not a shell: subprocess passes arguments verbatim.
    args=[arg.split("=",1)[0]+"='"+arg.split("=",1)[1]+"'" if "contig=" in arg else arg for arg in args]
    for key,value in c.get("sampler_overrides",{}).items():
        args=[arg for arg in args if not arg.startswith("inference_sampler."+key+"=")]
        args.append(f"inference_sampler.{key}={value}")
    for i in (1,2):
        args += [f"+track_{i}_specification.select_fixed_atoms=false",
                 f"+track_{i}_specification.select_unfixed_sequence=true",
                 f"+track_{i}_specification.is_non_loopy={str(c.get('non_loopy',True)).lower()}"]
    # A file avoids Hydra ambiguities for nested schedule dictionaries.
    experiment=case_dir(run,row)/"experiment.json"
    write(experiment,cfg,exclusive=True)
    args += ["+experiment_file="+container(experiment)]
    command(foundry(args,gpu=True))
    tracks={}
    for i in (1,2):
        paths=list(out.glob(f"*_track{i}_model_0.cif.gz"))
        if len(paths)!=1:
            raise RuntimeError(f"expected one track {i} structure, found {paths}")
        tracks[f"track{i}"]=str(paths[0])
    write(out/"_COMPLETE.json",tracks|dict(seed=row["seed"],condition=c["id"]))


def prepare_caliby(run,row):
    parent=case_dir(run,row)
    require_complete(parent/"rfd")
    out=parent/"caliby"
    if (out/"tied_caliby_manifest.json").exists():
        return
    command(foundry(["/project/software/foundry/models/rfd3_system/scripts/build_tied_mpnn_input.py",
        container(parent/"rfd"),"--out-dir",container(out),"--prepare-for","caliby",
        "--shared-chain-source-mode","per-track","--fixed-a-source-residues","",
        "--fixed-b-source-residues","","--translation-distance","100"]))


def caliby(run,row):
    import random
    import numpy as np
    import torch
    import pandas as pd
    from caliby import load_model
    parent=case_dir(run,row)
    out=parent/"caliby"
    if (out/"_COMPLETE.json").exists():
        return
    manifest=load(out/"tied_caliby_manifest.json")
    entries=manifest["entries"]
    if len(entries)!=1:
        raise ValueError("one paired backbone per task required")
    entry=entries[0]
    structure=host(entry["combined_pdb"])
    constraints=pd.DataFrame([dict(pdb_key=structure.stem,fixed_pos_seq="",
        symmetry_pos="|".join(",".join(group) for group in entry["symmetry_residues"]))])
    constraints.to_csv(out/"caliby_constraints.csv",index=False)
    s=seed("caliby",row["seed"])
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    model=load_model("soluble_caliby_v1",device="cpu")
    results=model.sample([str(structure)],out_dir=str(out/"designed"),num_seqs_per_pdb=4,
        batch_size=4,num_workers=0,temperature=.01,omit_aas=["C"],pos_constraint_df=constraints,verbose=True)
    pd.DataFrame(results).to_csv(out/"seq_des_outputs.csv",index=False)
    sys.path.insert(0,str(HELPERS))
    from sequence_design_io import load_sequence_design_manifest,load_sequence_design_records
    records=load_sequence_design_records(out,load_sequence_design_manifest(out,"caliby"),"caliby")
    if len(records)!=4 or any(r.chains["A"]!=r.chains["D"] for r in records):
        raise AssertionError("Caliby did not produce four tied A sequences")
    write(out/"_COMPLETE.json",dict(seed=s,candidates=4))


def fold(run,row):
    parent=case_dir(run,row)
    require_complete(parent/"caliby")
    out=parent/"folds"
    if (out/"_COMPLETE.json").exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(HELPERS))
    from sequence_design_io import load_sequence_design_manifest,load_sequence_design_records
    from fold_tied_mpnn_designability import build_designability_tasks
    from fold_mpnn_esmfold2 import load_esmfold2_model,run_fold,task_is_complete
    records=load_sequence_design_records(parent/"caliby",load_sequence_design_manifest(parent/"caliby","caliby"),"caliby")
    tasks=build_designability_tasks(records)
    repo,model=load_esmfold2_model("esmfold2")
    results=[]
    for task in tasks:
        if task_is_complete(out,task.task_id):
            results.append(load(out/(task.task_id+".json")))
            continue
        s=seed("fold",row["seed"],task.record.design_index,task.complex_kind)
        result=run_fold(task,model,"esmfold2",repo,out,dict(seed=s,num_diffusion_samples=1))
        results.append(result)
    write(out/"_COMPLETE.json",dict(folds=results,count=len(results)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["generate","prepare_caliby","caliby","fold"])
    p.add_argument("run")
    p.add_argument("--index",type=int,default=int(os.environ.get("SLURM_ARRAY_TASK_ID","0")))
    a=p.parse_args()
    run=Path(a.run).resolve()
    manifest=load(run/"manifest.json")
    verify_manifest(manifest)
    row=manifest["rows"][a.index]
    parent=case_dir(run,row)
    parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    try:
        globals()[a.stage](run,row)
        if not (parent/(a.stage+"_job.json")).exists():
            write(parent/(a.stage+"_job.json"),dict(job=os.environ.get("SLURM_JOB_ID"),seconds=time.monotonic()-start,
                source=manifest["source_revision"],seed=row["seed"],condition=row["condition"]["id"]))
    except Exception as exc:
        write(parent/(a.stage+"_FAILED.json"),dict(error=repr(exc),job=os.environ.get("SLURM_JOB_ID")))
        raise


if __name__=="__main__":
    main()
