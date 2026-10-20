import argparse, json, os
import numpy as np
from engine import Tensor, softmax_cross_entropy
from nn import MLP, Adam
from prune import GlobalPruner, cubic_sparsity
from train.data import make_moons
from train.common import train_epoch,evaluate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--sparsity',type=float,default=.8); ap.add_argument('--criterion',choices=['saliency','magnitude'],default='saliency'); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--out',default='results/self_prune.json'); args=ap.parse_args()
    x,y,xt,yt=make_moons(seed=args.seed); model=MLP([2,32,32,2],seed=args.seed); opt=Adam(model.parameters(),lr=.01); pruner=GlobalPruner(model,opt); rng=np.random.default_rng(args.seed+100)
    hist=[]; warmup=max(2,args.epochs//10); prune_end=max(warmup+1,args.epochs*3//4)
    for ep in range(1,args.epochs+1):
        tr=train_epoch(model,opt,x,y,64,rng)
        if ep>=warmup and ep<=prune_end:
            target=cubic_sparsity(ep-warmup,prune_end-warmup,args.sparsity)
            # Recompute gradients on the full training set for a lower-variance saliency estimate.
            for param in model.parameters(): param.zero_grad()
            full_loss=softmax_cross_entropy(model(Tensor(x)),y); full_loss.backward()
            grads=[layer.weight.grad.copy() for layer in model.layers]
            pruner.prune_to_sparsity(target,args.criterion,grads)
        vl,va=evaluate(model,xt,yt)
        hist.append({'epoch':ep,'train_loss':tr,'test_loss':vl,'test_accuracy':va,'achieved_sparsity':1-model.active_weight_count()/model.total_weight_count()})
        if ep==1 or ep%10==0 or ep==args.epochs: print(f"epoch={ep:03d} loss={vl:.4f} acc={va:.4f} sparsity={hist[-1]['achieved_sparsity']:.3f}")
    # Restore exact zeros after final optimizer step and measure sparse-aware forward consistency.
    for layer in model.layers: layer.weight.data[~layer.mask]=0
    sparse_logits=model.predict_sparse(xt); sparse_acc=float((sparse_logits.argmax(axis=1)==yt).mean())
    result={'seed':args.seed,'criterion':args.criterion,'target_sparsity':args.sparsity,'achieved_sparsity':1-model.active_weight_count()/model.total_weight_count(),'active_weights':model.active_weight_count(),'total_weights':model.total_weight_count(),'test_loss':evaluate(model,xt,yt)[0],'test_accuracy':evaluate(model,xt,yt)[1],'sparse_path_accuracy':sparse_acc,'history':hist}
    os.makedirs(os.path.dirname(args.out) or '.',exist_ok=True)
    with open(args.out,'w') as f: json.dump(result,f,indent=2)
if __name__=='__main__': main()
