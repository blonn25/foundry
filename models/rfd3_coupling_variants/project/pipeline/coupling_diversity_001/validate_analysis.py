"""Exercise real file parsing, topology, and exhaustive Foldseek contracts."""
import argparse
import numpy as np
from common import *
from metrics import *
from diversity import tm_matrix


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("validation",type=Path);p.add_argument("out",type=Path);a=p.parse_args()
    entries=[]
    for i,state in [(1,"AB"),(2,"AC")]:
        path=a.validation/"seed_90022"/f"design_0_track{i}_model_0.cif.gz"
        model=structure(path)
        assert len(topology(model,"A"))==90
        assert coordinates(model,"A",expected=90).shape==(90,3)
        entries.append((f"s1_{state}",path))
    tm,coverage=tm_matrix(a.out,entries)
    assert np.allclose(tm,1,atol=.001),(tm,coverage)
    write(a.out/"_COMPLETE.json",dict(tm=tm.tolist(),coverage=coverage.tolist()))
