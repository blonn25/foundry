"""Project paths, immutable records, and reproducible seed derivation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/"software/foundry/models/rfd3_coupling_variants"
HELPERS=ROOT/"software/foundry/models/rfd3_system/scripts"
sys.path.insert(0,str(BUNDLE/"src"))


def seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str,(20261004,)+parts)).encode()).digest()[:4],"big") % (2**31-1)


def load(path):
    return json.loads(Path(path).read_text())


def write(path,value,exclusive=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open("x") as handle:
            json.dump(value,handle,indent=2,allow_nan=False)
        return
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temp,path)


def revision():
    return subprocess.check_output(["git","-C",str(ROOT/"software/foundry"),"rev-parse","HEAD"],text=True).strip()


def controls():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}


def verify_manifest(manifest):
    if revision()!=manifest["source_revision"]:
        raise RuntimeError("source revision differs from immutable manifest")
    if manifest.get("pipeline_sha256") is not None and controls()!=manifest["pipeline_sha256"]:
        raise RuntimeError("pipeline controls differ from immutable manifest")


def command(args):
    print("COMMAND",repr([str(a) for a in args]),flush=True)
    subprocess.run([str(a) for a in args],cwd=ROOT,check=True)


def container(path):
    return "/project/"+str(Path(path).resolve().relative_to(ROOT))


def host(path):
    return ROOT/str(path).removeprefix("/project/") if str(path).startswith("/project/") else Path(path)


def foundry(args,gpu=False):
    return [str(ROOT/"scripts/foundry_exec.sh")]+(["--gpu"] if gpu else [])+[
        "env","PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=/project/software/foundry/models/rfd3_coupling_variants/src:/project/software/foundry/models/rfd3_system/src:/project/software/foundry/src",
        "python"]+list(map(str,args))


def case_dir(run,row):
    return Path(run)/row["condition"]["id"]/f"seed_{row['seed']}"


def require_complete(path):
    marker=Path(path)/"_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"missing successful stage marker {marker}")
    return load(marker)
