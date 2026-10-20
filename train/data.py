import numpy as np

def make_moons(n=1200, noise=0.12, seed=7):
    rng=np.random.default_rng(seed); n0=n//2; n1=n-n0
    t0=rng.uniform(0,np.pi,n0); t1=rng.uniform(0,np.pi,n1)
    a=np.column_stack([np.cos(t0),np.sin(t0)])
    b=np.column_stack([1-np.cos(t1),0.5-np.sin(t1)])
    x=np.vstack([a,b])+rng.normal(0,noise,(n,2)); y=np.r_[np.zeros(n0,dtype=np.int64),np.ones(n1,dtype=np.int64)]
    idx=rng.permutation(n); x=x[idx]; y=y[idx]
    x=(x-x.mean(axis=0))/(x.std(axis=0)+1e-8)
    split=int(0.8*n)
    return x[:split],y[:split],x[split:],y[split:]
