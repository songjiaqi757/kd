"""Small tests of grouping, transform, ridge invariance, and metric boundaries."""
import numpy as np
from analyze import groupsplits,ridge,choose,quad,ranked,matrix
from common import *

rng=np.random.default_rng(13);X=rng.normal(size=(90,7));y=2*X[:,0]-.5*X[:,3]+.1*rng.normal(size=90);g=np.repeat(np.arange(15),6)
for tr,va in groupsplits(g,5):
    assert not set(g[tr])&set(g[va])
    a=choose(X[tr],y[tr],g[tr],'response');p=ridge(X[tr],y[tr],X[va],a)
    # A test-only shift cannot influence training standardization/fit.
    shifted=ridge(X[tr]+17,y[tr],X[va]+17,a)
    np.testing.assert_allclose(p,shifted,atol=1e-10)
assert np.mean((ridge(X,y,X,.01)-y)**2)<.03
np.testing.assert_allclose(ridge(X[:,:0],y,X[:5,:0],0),np.repeat(y.mean(),5))
M=matrix();baseline=.23;coords=(X-baseline)@M.T
np.testing.assert_allclose(baseline+coords@np.linalg.inv(M).T,X,atol=1e-12)
assert quad(np.array([0.,1.,-1.]),np.array([0.,1.,1.]),np.array([1.,-1.,-1.]),'mosi').tolist()==['Q0_zero_excluded','Q2','Q3']
assert quad(np.array([0.]),np.array([0.]),np.array([0.]),'sims').tolist()==['Q1']
np.testing.assert_allclose(ranked(np.array([2,1,2])),[1.5,0,1.5])
save(OUT/'status/preflight.json',dict(status='complete',tests=['source_group_disjointness','training_only_scaling_translation_invariance','ridge_known_linear_signal','empty_baseline','Mobius_inverse','official_Acc2_zero_rules','Spearman_tie_ranks'],parameter_updates=0))
print('Phase A preflight passed')
