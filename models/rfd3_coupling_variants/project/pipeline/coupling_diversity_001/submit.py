"""Static matching SLURM arrays, immutable inputs, and explicit completion gates."""
import argparse
import fcntl
from common import *


def submit(run,validation=None,smoke=None,indices=None):
    run=Path(run).resolve();manifest=load(run/"manifest.json")
    verify_manifest(manifest)
    if manifest["stage"] not in ("smoke","calibration"):
        if not validation or not smoke:raise ValueError("validated checkpoint and full workflow smoke paths required")
        require_complete(validation);require_complete(smoke)
        calibration=load(manifest["calibration"]) if manifest.get("calibration") else None
        if manifest["stage"]=="screen" and not calibration:raise ValueError("frozen calibration required")
    lock=(run/".submit.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record=run/("submission.json" if indices is None else "recovery_"+indices.replace(",","_")+".json")
    if record.exists():raise FileExistsError(f"already submitted: {record}; inspect existing jobs before recovery")
    # Journal each accepted job immediately so a later submission error never
    # hides a live predecessor. Sparse recoveries must be explicitly requested.
    jobs={}
    def sbatch(stage,dependency=None,array=False):
        args=["sbatch","--parsable","--kill-on-invalid-dep=yes"]
        if dependency:args += ["--dependency="+dependency]
        if array:args += ["--array="+(indices or f"0-{len(manifest['rows'])-1}")]
        args += [str(ROOT/f"jobs/coupling_diversity_{stage}.sbatch"),str(run)]
        if stage=="analysis" and manifest["stage"]=="calibration":args.append("calibration")
        job=subprocess.check_output(args,cwd=ROOT,text=True).strip().split(";")[0]
        jobs[stage]=job;write(record,dict(jobs=jobs,source_revision=revision(),indices=indices))
        return job
    gen=sbatch("generate",array=True)
    if manifest["stage"]=="calibration":
        sbatch("analysis","afterok:"+gen)
    else:
        cal=sbatch("caliby","aftercorr:"+gen,array=True)
        folds=sbatch("fold","aftercorr:"+cal,array=True)
        sbatch("analysis","afterok:"+folds)
    print(json.dumps(jobs,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run");p.add_argument("--validation");p.add_argument("--smoke")
    p.add_argument("--indices",help="explicit sparse recovery indices, after inspecting previous failures")
    a=p.parse_args();submit(a.run,a.validation,a.smoke,a.indices)
