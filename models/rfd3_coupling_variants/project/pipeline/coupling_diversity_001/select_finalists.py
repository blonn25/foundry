"""Evidence-based shortlist; never shadow Python's standard-library select module."""
import argparse
import numpy as np
from common import *


def select(run):
    stats={s["condition"]:s for s in load(run/"analysis/conditions.json")}
    div={s["condition"]:s for s in load(run/"analysis/diversity.json")}
    manifest=load(run/"manifest.json");base=stats["mean_5050"]
    if any(s["complete"]!=s["attempted"] for s in stats.values()):
        raise ValueError("incomplete screen cannot select finalists")
    candidates=[];audit=[]
    def diversity(name):
        means=[div[name]["states"][state]["successful"]["TM_mean"] for state in ("AB","AC")]
        return None if any(v is None for v in means) else min(1-v for v in means)
    baseline_diversity=diversity("mean_5050")
    for name,s in stats.items():
        if name=="mean_5050" or s["model"]=="rfd3_mean_5050":continue
        d=diversity(name)
        quality=s["dual_success_rate"]>=base["dual_success_rate"]
        # An undefined baseline diversity cannot establish a diversity gain.
        # Positive-yield families may still warrant confirmation, labelled below.
        gain=d-baseline_diversity if d is not None and baseline_diversity is not None else None
        eligible=quality and d is not None and (gain is None or gain>0)
        row=dict(condition=name,model=s["model"],eligible=eligible,dual_success_rate=s["dual_success_rate"],
            successful_diversity=d,gain=gain,baseline_diversity_defined=baseline_diversity is not None)
        audit.append(row)
        if eligible:candidates.append(row)
    candidates.sort(key=lambda r:(r["gain"] if r["gain"] is not None else r["dual_success_rate"],r["dual_success_rate"]),reverse=True)
    chosen=[];families=set()
    for row in candidates:
        if row["model"] in families:continue
        chosen.append(row);families.add(row["model"])
        if len(chosen)==3:break
    names={r["condition"] for r in chosen}|{"mean_5050"}
    payload=dict(conditions=[c for c in manifest["conditions"] if c["id"] in names],audit=audit,
        selected=chosen,status="confirmation candidates, not established winners" if chosen else "no qualifying alternative",
        selection="observed dual success at least baseline; positive worst-state successful-pool TM diversity gain; at most three distinct families",
        caveat="If fewer than two baseline parents pass, baseline diversity is undefined; selection prioritizes positive-yield methods without claiming a diversity gain.")
    write(run/"analysis/selected.json",payload)
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);a=p.parse_args();print(json.dumps(select(a.run),indent=2))
