"""Publication/export plots and a compact report, generated from recorded metrics."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from collect import coupling


def save(fig,directory,name):
    fig.savefig(directory/(name+".png"),dpi=180,bbox_inches="tight")
    fig.savefig(directory/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)


def plot(run):
    stats=load(run/"analysis/conditions.json");parents=load(run/"analysis/parents.json")
    diversity={r["condition"]:r for r in load(run/"analysis/diversity.json")}
    manifest=load(run/"manifest.json");out=run/"analysis/plots";out.mkdir(exist_ok=True)
    valid=[s for s in stats if s["complete"]]
    colors=plt.get_cmap("tab20")
    for scope in ("all","successful"):
        fig,ax=plt.subplots(figsize=(9,6))
        for i,s in enumerate(valid):
            d=diversity[s["condition"]]
            ys=[d["states"][st][scope]["TM_mean"] for st in ("AB","AC")]
            if any(y is None for y in ys):continue
            y=min(1-v for v in ys)
            ax.scatter(s["dual_success_rate"],y,color=colors(i%20),marker="*" if s["condition"]=="mean_5050" else "o")
            ax.annotate(s["condition"],(s["dual_success_rate"],y),fontsize=6,xytext=(3,3),textcoords="offset points")
        ax.set(xlabel="Dual-state backbone success rate",ylabel=f"Minimum state diversity (1 − mean TM-score), {scope}",
               title="Compatibility–diversity comparison")
        ax.grid(alpha=.2);save(fig,out,"frontier_"+scope)
    fig,axs=plt.subplots(1,2,figsize=(12,max(3,len(valid)*.23)),sharey=True)
    y=np.arange(len(valid))
    for ax,state in zip(axs,("AB","AC")):
        values=[s[state+"_success_rate"] for s in valid]
        err=np.array([[v-s[state+"_success_ci"][0],s[state+"_success_ci"][1]-v] for s,v in zip(valid,values)]).T
        ax.errorbar(values,y,xerr=err,fmt="o",capsize=2)
        ax.set(xlim=(-.03,1.03),xlabel="Backbone success rate (95% Wilson interval)",title="A+"+state[-1])
    axs[0].set_yticks(y,[s["condition"] for s in valid]);save(fig,out,"state_success")
    baseline=next((s["rfd_seconds_mean"] for s in valid if s["condition"]=="mean_5050"),None)
    if baseline:
        fig,ax=plt.subplots(figsize=(10,max(3,len(valid)*.23)))
        ax.barh(y,[s["rfd_seconds_mean"]/baseline for s in valid]);ax.set_yticks(y,[s["condition"] for s in valid])
        ax.set(xlabel="Mean RFD sampler runtime relative to 50/50");save(fig,out,"runtime")
    # Lightweight per-step traces cover all samples, while full coordinate dumps
    # are restricted to the first three seeds per condition.
    for xkey,name in [("completed_fraction","progress"),("sigma","sigma")]:
        fig,ax=plt.subplots(figsize=(11,6))
        for i,s in enumerate(valid):
            traces=[]
            for row in manifest["rows"]:
                if row["condition"]["id"]==s["condition"]:
                    traces.append(coupling(case_dir(run,row))["trace"])
            x=[t[xkey] for t in traces[0]];values=np.array([[t["ca_rmsd"] for t in tr] for tr in traces])
            ax.plot(x,values.mean(0),label=s["condition"],color=colors(i%20),lw=1)
        if xkey=="sigma":ax.set_xscale("log");ax.invert_xaxis()
        ax.set(xlabel="Denoising fraction" if xkey=="completed_fraction" else "Sigma (Å)",ylabel="Mean aligned A-state Cα RMSD (Å)")
        ax.legend(fontsize=6,ncol=2,bbox_to_anchor=(1.02,1),loc="upper left");save(fig,out,"trajectory_"+name)
    for s in valid:
        row=next(r for r in manifest["rows"] if r["condition"]["id"]==s["condition"])
        tr=coupling(case_dir(run,row))["trace"][1:];x=[t["progress"] for t in tr]
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        axs[0,0].plot(x,[t["ca_rmsd"] for t in tr]);axs[0,0].set_ylabel("Aligned A Cα RMSD (Å)")
        axs[0,1].plot(x,[t["js"] for t in tr]);axs[0,1].set_ylabel("Native sequence JS")
        for i,state in enumerate(("A+B","A+C")):
            axs[1,0].plot(x,[t["sequence_entropy"][i] for t in tr],label=state)
            axs[1,1].plot(x,[t.get("total_A_correction_ratios",[r*t["cap_factor"] for r in t["raw_correction_ratios"]])[i] for t in tr],label=state)
        axs[1,0].set_ylabel("Native sequence entropy (nats)");axs[1,1].set_ylabel("Total A correction / native update norm")
        axs[1,0].legend();axs[1,1].legend()
        for ax in axs.flat:ax.set_xlabel("Denoising fraction")
        fig.suptitle(s["condition"]+f" — seed {row['seed']}");fig.tight_layout();save(fig,out,"diagnostics_"+s["condition"])
    lines=["# Coupling diversity experiment", "",f"Stage: {manifest['stage']}. Source: `{manifest['source_revision']}`.","",
        "Success requires the same tied Caliby candidate to have whole-complex Cα RMSD <2 Å in both states. Confidence is diagnostic. No FastRelax is used.","",
        "| Condition | Complete pairs | A+B | A+C | Dual | A RMSD (Å) | A TM | Successful clusters AB / AC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in valid:
        d=diversity[s["condition"]];counts=[d["states"][st]["successful"]["clusters"] for st in ("AB","AC")]
        lines.append(f"| {s['condition']} | {s['complete']} | {s['AB_success_rate']:.1%} | {s['AC_success_rate']:.1%} | {s['dual_success_rate']:.1%} | {s['A_ca_rmsd_mean']:.3f} | {d['within_pair_TM_mean']:.3f} | {counts[0]} / {counts[1]} |")
    lines += ["", "Diversity is reported for all parents and dual-state passers separately. Cluster counts use complete linkage, TM ≥0.5 and ≥80% alignment coverage; 0.6/0.7 sensitivity and sample-count rarefaction are in diversity.json.",
        "", "These are computational designability results, not experimental binding validation. Candidate-level outcomes, state-specific confidence and interface geometry are retained in the CSV/JSON files."]
    (run/"analysis/report.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);a=p.parse_args();plot(a.run)
