"""Restore missing project controls; refuse differing existing files."""
import argparse
from pathlib import Path
import shutil


def install(root):
    source=Path(__file__).resolve().parent/"project"
    files=[p for p in source.rglob("*") if p.is_file()]
    conflicts=[str(p.relative_to(source)) for p in files if (root/p.relative_to(source)).exists()
               and (root/p.relative_to(source)).read_bytes()!=p.read_bytes()]
    if conflicts:raise FileExistsError("existing project controls differ: "+", ".join(conflicts))
    for path in files:
        target=root/path.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            print("restored",target)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("project_root",type=Path)
    args=parser.parse_args();install(args.project_root.resolve())
