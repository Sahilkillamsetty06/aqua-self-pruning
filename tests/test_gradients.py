import unittest
import numpy as np
from engine import Tensor, softmax_cross_entropy

def numeric_grad(fn, x, eps=1e-6):
    g=np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old=x[idx]; x[idx]=old+eps; a=fn(x); x[idx]=old-eps; b=fn(x); x[idx]=old; g[idx]=(a-b)/(2*eps)
    return g

class GradientTests(unittest.TestCase):
    def check(self, fn, arrays, build, atol=2e-5):
        tensors=[Tensor(a.copy(),True) for a in arrays]; out=build(*tensors); out.backward()
        for i,a in enumerate(arrays):
            ng=numeric_grad(lambda z: float(fn(*[z if j==i else arrays[j] for j in range(len(arrays))])),a.copy())
            np.testing.assert_allclose(tensors[i].grad,ng,atol=atol,rtol=atol)
    def test_add_broadcast(self):
        a=np.random.default_rng(1).normal(size=(2,3)); b=np.random.default_rng(2).normal(size=(3,))
        self.check(lambda x,y: np.sum(x+y),[a,b],lambda x,y:(x+y).sum())
    def test_subtract(self):
        a=np.array([[1.2,-.5],[.7,2.]]); b=np.array([.4,1.1])
        self.check(lambda x,y: np.sum(x-y),[a,b],lambda x,y:(x-y).sum())
    def test_multiply_broadcast(self):
        a=np.array([[1.,2.,3.],[4.,5.,6.]]); b=np.array([.2,.3,.4])
        self.check(lambda x,y: np.sum(x*y),[a,b],lambda x,y:(x*y).sum())
    def test_divide(self):
        a=np.array([[1.2,2.],[.7,3.]]); b=np.array([1.4,2.2])
        self.check(lambda x,y: np.sum(x/y),[a,b],lambda x,y:(x/y).sum())
    def test_matmul(self):
        a=np.array([[.2,.4],[-.3,.8]]); b=np.array([[.5,-.7],[1.2,.1]])
        self.check(lambda x,y: np.sum(x@y),[a,b],lambda x,y:(x@y).sum())
    def test_sum_mean(self):
        a=np.array([[.2,.4],[-.3,.8]])
        self.check(lambda x: np.mean(x,axis=0).sum(),[a],lambda x:x.mean(axis=0).sum())
    def test_relu(self):
        a=np.array([[-.8,.4],[.2,1.3]])
        self.check(lambda x: np.maximum(x,0).sum(),[a],lambda x:x.relu().sum())
    def test_tanh(self):
        a=np.array([[-.8,.4],[.2,1.3]])
        self.check(lambda x: np.tanh(x).sum(),[a],lambda x:x.tanh().sum())
    def test_sigmoid(self):
        a=np.array([[-.8,.4],[.2,1.3]])
        self.check(lambda x: (1/(1+np.exp(-x))).sum(),[a],lambda x:x.sigmoid().sum())
    def test_softmax_cross_entropy(self):
        a=np.array([[.2,.4,-.1],[1.2,-.7,.3]]) ; y=np.array([2,0])
        def num(x):
            z=x-x.max(axis=1,keepdims=True); return float((-z[np.arange(len(y)),y]+np.log(np.exp(z).sum(axis=1))).mean())
        t=Tensor(a.copy(),True); softmax_cross_entropy(t,y).backward(); ng=numeric_grad(num,a.copy())
        np.testing.assert_allclose(t.grad,ng,atol=2e-5,rtol=2e-5)

if __name__=='__main__': unittest.main()
