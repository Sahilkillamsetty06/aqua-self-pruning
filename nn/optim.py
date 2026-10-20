import numpy as np

class Adam:
    """Adam with explicit state reset for every pruned or revived parameter."""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params); self.lr=lr; self.beta1=beta1; self.beta2=beta2; self.eps=eps; self.t=0
        self.m = {id(p): np.zeros_like(p.data) for p in self.params}
        self.v = {id(p): np.zeros_like(p.data) for p in self.params}
    def step(self, masks_by_param=None):
        self.t += 1
        masks_by_param = masks_by_param or {}
        for p in self.params:
            if not p.requires_grad: continue
            key=id(p); g=p.grad
            mask=masks_by_param.get(key)
            if mask is not None:
                # No gradient or stale momentum may update inactive coordinates.
                g = np.where(mask, g, 0.0)
                self.m[key] = np.where(mask, self.m[key], 0.0)
                self.v[key] = np.where(mask, self.v[key], 0.0)
            self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*g
            self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(g*g)
            mhat=self.m[key]/(1-self.beta1**self.t); vhat=self.v[key]/(1-self.beta2**self.t)
            update=self.lr*mhat/(np.sqrt(vhat)+self.eps)
            if mask is not None: update=np.where(mask, update, 0.0)
            p.data -= update
            if mask is not None: p.data[~mask]=0.0
    def on_mask_change(self, param, changed_mask):
        """Reset Adam moments at coordinates whose connectivity changed."""
        key=id(param)
        self.m[key][changed_mask]=0.0
        self.v[key][changed_mask]=0.0
