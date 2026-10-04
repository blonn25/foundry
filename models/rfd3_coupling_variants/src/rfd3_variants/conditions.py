"""Frozen 37-condition, one-factor screening matrix (stdlib only)."""
from copy import deepcopy


def ramp(start,end,kind="linear",until=1.):
    return dict(start=start,end=end,kind=kind,until=until)


def conditions():
    rows=[]
    def add(name,method,**experiment):
        row=dict(id=name,model="rfd3_"+method,experiment=experiment)
        rows.append(row)
        return row
    add("mean_5050","mean_5050")
    add("uncoupled_shared","uncoupled")
    add("uncoupled_independent","uncoupled",shared_initialization=False,noise_correlation=0.)
    for f in (.5,.7,.8,.9,.95):
        add(f"late_{f:.2f}","late_uncoupling",release_fraction=f)
    for a in (.1,.25,.5,.75):
        add(f"residual_{a:g}","residual_consensus",alpha=a)
    for kind in ("linear","cosine"):
        for a in (0.,.1,.25):
            add(f"residual_{kind}_{a:g}_1","residual_consensus",alpha=ramp(a,1.,kind))
    for rho in (.75,.5):
        add(f"noise_{rho:g}","correlated_noise",noise_correlation=rho)
    for rho in (1.,.75):
        add(f"noise_{rho:g}_0","correlated_noise",noise_correlation=ramp(rho,0.))
    for ratio in (.03,.1,.3):
        row=add(f"soft_cosine_{ratio:g}","soft_guidance",guidance_schedule=ramp(1.,0.,"cosine"))
        row["calibration_target"]=ratio
    for name,target,sched in [("constant",.03,1.),("linear",.1,ramp(1.,0.)),("cut_080",.1,ramp(1.,0.,"cosine",.8))]:
        row=add("soft_"+name,"soft_guidance",guidance_schedule=sched)
        row["calibration_target"]=target
    for block in (3,5,9):
        row=add(f"coarse_{block}","coarse_coupling",block=block,guidance_schedule=ramp(1.,0.,"cosine"))
        row["calibration_target"]=.1
    for ratio in (.03,.1,.3):
        row=add(f"sequence_{ratio:g}","sequence_coupling",temperature=ramp(2.,1.))
        row["calibration_target"]=ratio
    row=add("sequence_T1","sequence_coupling",temperature=1.)
    row["calibration_target"]=.1
    row["calibration_reference"]="sequence_coupling"
    add("mean_step_1p5","mean_5050")["sampler_overrides"]={"step_scale":1.5}
    add("mean_without_nonloopy","mean_5050")["non_loopy"]=False
    if len(rows)!=37 or len({r["id"] for r in rows})!=37:
        raise AssertionError("screening matrix contract violated")
    return deepcopy(rows)


def calibration_key(row):
    method=row["model"].removeprefix("rfd3_")
    return f"coarse_{row['experiment']['block']}" if method=="coarse_coupling" else method
