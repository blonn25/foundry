"""Scientific contracts exercised on CPU through SLURM."""
import unittest
import numpy as np
from metrics import rmsd,summarize_candidates
from diversity import clusters,diverse_representatives
from rfd3_variants.conditions import conditions


class Contracts(unittest.TestCase):
    def test_same_candidate_required(self):
        outcome=summarize_candidates([dict(AB_pass=True,AC_pass=False),dict(AB_pass=False,AC_pass=True)])
        self.assertFalse(outcome["dual_pass"])
        self.assertTrue(outcome["AB_pass"] and outcome["AC_pass"])

    def test_rigid_motion_irrelevant(self):
        x=np.random.default_rng(4).normal(size=(20,3));q,_=np.linalg.qr(np.random.default_rng(5).normal(size=(3,3)))
        self.assertLess(rmsd(x,x@q+15),1e-12)

    def test_reflection_not_removed(self):
        x=np.random.default_rng(6).normal(size=(20,3));y=x.copy();y[:,0]*=-1
        self.assertGreater(rmsd(x,y),.1)

    def test_complete_link_not_single_link(self):
        t=np.array([[1,.8,.2],[.8,1,.8],[.2,.8,1]])
        self.assertEqual(len(set(clusters(t))),2)

    def test_population_distinct_representatives(self):
        x=np.arange(20)[:,None].astype(float)
        ids=diverse_representatives(x,10)
        self.assertEqual(len(set(ids)),10)

    def test_screen_counts(self):
        rows=conditions();self.assertEqual(len(rows),37)
        self.assertEqual(len(rows)*50*4*2,14800)


if __name__=="__main__":unittest.main()
