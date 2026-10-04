"""Exhaustive per-condition TM-align, clustering, rarefaction and population selection."""
import argparse
import re
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform,pdist
from common import *
from metrics import structure,coordinates,rmsd


def clusters(tm,threshold=.5):
    if len(tm)<2:return np.ones(len(tm),int)
    return fcluster(linkage(squareform(np.clip(1-tm,0,1),checks=False),method="complete"),
                    1-threshold,criterion="distance")


def cluster_stats(labels):
    if not len(labels):return dict(clusters=0,effective_clusters=0.)
    p=np.array(list(Counter(labels).values()),float)/len(labels)
    return dict(clusters=len(p),effective_clusters=float(np.exp(-(p*np.log(p)).sum())))


def diverse_representatives(descriptors,count):
    if not len(descriptors):return []
    # Start with the most central member, then maximize distance to the bank.
    selected=[int(np.argmin(((descriptors-descriptors.mean(0))**2).sum(1)))]
    distance=np.full(len(descriptors),np.inf)
    while len(selected)<min(count,len(descriptors)):
        distance=np.minimum(distance,((descriptors-descriptors[selected[-1]])**2).mean(1))
        distance[selected]=-1
        selected.append(int(np.argmax(distance)))
    return selected


def rarefaction(labels,rng):
    result=[]
    for k in sorted(set([1,5,10,20,50,len(labels)])):
        if k>len(labels) or not k:continue
        counts=[len(set(rng.choice(labels,k,replace=False))) for _ in range(200)]
        result.append(dict(n=k,mean=float(np.mean(counts)),low=float(np.quantile(counts,.025)),high=float(np.quantile(counts,.975))))
    return result


def tm_matrix(directory,entries):
    """No prefilter shortcuts: require every directed comparison, including self."""
    from Bio.PDB import PDBIO,Select
    class OnlyA(Select):
        def accept_chain(self,chain):return chain.id=="A"
    source=directory/"structures";source.mkdir(parents=True,exist_ok=True)
    for label,path in entries:
        io=PDBIO();io.set_structure(structure(host(path)));io.save(str(source/(label+".pdb")),OnlyA())
    result=directory/"tmalign.tsv"
    if not result.exists():
        command([ROOT/"software/foldseek/bin/foldseek","easy-search",source,source,result,directory/"tmp",
            "--alignment-type","1","--exhaustive-search","1","--add-self-matches","1",
            "--max-seqs",str(len(entries)),"-e","1e9","--threads",os.environ.get("SLURM_CPUS_PER_TASK","8"),
            "--format-output","query,target,qtmscore,ttmscore,alnlen,qlen,tlen"])
    index={label:i for i,(label,_) in enumerate(entries)};n=len(entries)
    score=np.full((n,n),np.nan);coverage=np.full((n,n),np.nan)
    def key(value):
        match=re.search(r"s\d+_(?:AB|AC)",value)
        if match is None or match.group() not in index:raise ValueError(f"unrecognized Foldseek label {value}")
        return index[match.group()]
    for line in result.read_text().splitlines():
        q,t,qt,tt,aln,ql,tl=line.split("\t")
        i,j=key(q),key(t);score[i,j]=min(float(qt),float(tt));coverage[i,j]=float(aln)/max(float(ql),float(tl))
    if not np.isfinite(score).all():raise ValueError(f"incomplete exhaustive TM-align table: {np.isnan(score).sum()} missing comparisons")
    score=np.minimum(score,score.T);coverage=np.minimum(coverage,coverage.T)
    np.fill_diagonal(score,1.)
    return score,coverage


def analyze(run):
    parents=load(run/"analysis/parents.json");manifest=load(run/"manifest.json")
    output=[];population=[]
    for config in manifest["conditions"]:
        ps=sorted([p for p in parents if p["condition"]==config["id"]],key=lambda p:p["seed"])
        if not ps:continue
        directory=run/"analysis/diversity"/config["id"];directory.mkdir(parents=True,exist_ok=True)
        entries=[(f"s{p['seed']}_{state}",p[state+"_reference"]) for state in ("AB","AC") for p in ps]
        tm,coverage=tm_matrix(directory,entries);n=len(ps)
        xyz=[coordinates(structure(host(path)),"A",expected=90) for _,path in entries]
        rmsds=np.zeros_like(tm)
        for i in range(2*n):
            for j in range(i):rmsds[i,j]=rmsds[j,i]=rmsd(xyz[i],xyz[j])
        np.savez_compressed(directory/"pairwise.npz",tm=tm,coverage=coverage,rmsd=rmsds,labels=[e[0] for e in entries])
        passing=np.array([i for i,p in enumerate(ps) if p["dual_pass"]],int)
        item=dict(condition=config["id"],within_pair_TM_mean=float(np.diag(tm[:n,n:]).mean()),
                  within_pair_TM_median=float(np.median(np.diag(tm[:n,n:]))),states={})
        for state,offset in [("AB",0),("AC",n)]:
            item["states"][state]={}
            for scope,subset in [("all",np.arange(n)),("successful",passing)]:
                inds=offset+subset;t=tm[np.ix_(inds,inds)];cov=coverage[np.ix_(inds,inds)]
                distances=rmsds[np.ix_(inds,inds)];tri=np.triu_indices(len(subset),1)
                # Short-fragment matches cannot establish a shared full fold.
                ct=np.where(cov>=.8,t,0.);np.fill_diagonal(ct,1.)
                labels=clusters(ct)
                stats=dict(n=len(subset),TM_mean=float(t[tri].mean()) if len(subset)>1 else None,
                    RMSD_mean=float(distances[tri].mean()) if len(subset)>1 else None,
                    low_coverage_pairs=int((cov[tri]<.8).sum()),**cluster_stats(labels),
                    cluster_membership={str(ps[i]["seed"]):int(label) for i,label in zip(subset,labels)},
                    sensitivity={str(v):cluster_stats(clusters(ct,v)) for v in (.5,.6,.7)},
                    rarefaction=rarefaction(labels,np.random.default_rng(seed(config["id"],state,scope))))
                item["states"][state][scope]=stats
        # Population postselection is scored at identical counts and on successful
        # paired parents; four sequence candidates never become four samples.
        if len(passing):
            descriptor=np.array([np.concatenate([pdist(xyz[i][::5]),pdist(xyz[i+n][::5])])/10 for i in passing])
            rng=np.random.default_rng(seed("population",config["id"]))
            for count in (5,10,20):
                if count>len(passing):continue
                chosen=passing[diverse_representatives(descriptor,count)]
                def mean_d(indices):
                    vals=[]
                    for off in (0,n):
                        sub=tm[np.ix_(off+indices,off+indices)];vals.append(1-sub[np.triu_indices(count,1)].mean())
                    return float(min(vals))
                random_values=[mean_d(rng.choice(passing,count,replace=False)) for _ in range(200)]
                population.append(dict(model="rfd3_population_diversity",source_condition=config["id"],count=count,
                    seeds=[ps[i]["seed"] for i in chosen],diversity=mean_d(chosen),
                    random_mean=float(np.mean(random_values)),random_ci=np.quantile(random_values,[.025,.975]).tolist()))
        output.append(item)
    write(run/"analysis/diversity.json",output);write(run/"analysis/population.json",population)
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);a=p.parse_args();analyze(a.run)
