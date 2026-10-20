import numpy as np
from engine import Tensor, softmax_cross_entropy

def accuracy(logits, y): return float((np.argmax(logits,axis=1)==y).mean())
def batches(x,y,batch_size,rng):
    order=rng.permutation(len(x))
    for start in range(0,len(x),batch_size):
        ids=order[start:start+batch_size]; yield x[ids],y[ids]
def evaluate(model,x,y):
    logits=model(Tensor(x)).data
    shifted=logits-logits.max(axis=1,keepdims=True); logsum=np.log(np.exp(shifted).sum(axis=1))
    loss=float((-shifted[np.arange(len(y)),y]+logsum).mean())
    return loss,accuracy(logits,y)
def train_epoch(model,opt,x,y,batch_size,rng, masks=True):
    losses=[]
    maskmap={id(layer.weight):layer.mask for layer in model.layers}
    for xb,yb in batches(x,y,batch_size,rng):
        for p in model.parameters(): p.zero_grad()
        loss=softmax_cross_entropy(model(Tensor(xb)),yb)
        loss.backward(); opt.step(maskmap if masks else None); losses.append(float(loss.data))
    return float(np.mean(losses))
