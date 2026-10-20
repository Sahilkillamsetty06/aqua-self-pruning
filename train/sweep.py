import argparse, csv, os, time
import numpy as np
from engine import Tensor, softmax_cross_entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nn import MLP, Adam
from prune import GlobalPruner, cubic_sparsity
from train.data import make_moons
from train.common import train_epoch, evaluate

def run_one(seed, target, criterion, epochs):
    x,y,xt,yt=make_moons(seed=seed); model=MLP([2,32,32,2],seed=seed); opt=Adam(model.parameters(),lr=.01); pruner=GlobalPruner(model,opt); rng=np.random.default_rng(seed+100)
    warmup=max(2,epochs//10); prune_end=max(warmup+1,epochs*3//4)
    for ep in range(1,epochs+1):
        train_epoch(model,opt,x,y,64,rng)
        if target>0 and warmup<=ep<=prune_end:
            schedule=cubic_sparsity(ep-warmup,prune_end-warmup,target)
            for param in model.parameters(): param.zero_grad()
            full_loss=softmax_cross_entropy(model(Tensor(x)),y); full_loss.backward()
            grads=[layer.weight.grad.copy() for layer in model.layers]
            pruner.prune_to_sparsity(schedule,criterion,grads)
    loss,acc=evaluate(model,xt,yt)
    # Count connection-level MACs for one sample: dense path evaluates every weight;
    # sparse-aware path visits only active connections. This is an operation count,
    # not a claim of equivalent wall-clock speedup on NumPy/CPU hardware.
    total=model.total_weight_count(); active=model.active_weight_count()
    t0=time.perf_counter()
    for _ in range(100): model.predict_sparse(xt)
    sparse_ms=(time.perf_counter()-t0)*1000/100
    t0=time.perf_counter()
    for _ in range(100): dense_forward(model, xt)
    dense_ms=(time.perf_counter()-t0)*1000/100
    return {'seed':seed,'criterion':criterion,'target_sparsity':target,'achieved_sparsity':1-active/total,'accuracy':acc,'loss':loss,'active_weights':active,'total_weights':total,'dense_weight_macs_per_sample':total,'sparse_weight_macs_per_sample':active,'operation_reduction':1-active/total,'dense_batch_ms':dense_ms,'sparse_batch_ms':sparse_ms}

# Dense inference timing uses NumPy only, without constructing an autodiff graph.

def dense_forward(model, x):
    for layer in model.layers:
        x=x @ (layer.weight.data*layer.mask) + layer.bias.data
        if layer.activation=='relu': x=np.maximum(x,0)
        elif layer.activation=='tanh': x=np.tanh(x)
    return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--seeds',default='7,17,27'); ap.add_argument('--sparsities',default='0,.5,.75,.9,.95'); ap.add_argument('--outdir',default='results'); ap.add_argument('--plotdir',default='plots'); args=ap.parse_args()
    rows=[]; seeds=[int(s) for s in args.seeds.split(',')]; sparsities=[float(s) for s in args.sparsities.split(',')]
    os.makedirs(args.outdir,exist_ok=True); os.makedirs(args.plotdir,exist_ok=True)
    for target in sparsities:
        methods=['magnitude'] if target==0 else ['magnitude','saliency']
        for method in methods:
            for seed in seeds:
                row=run_one(seed,target,method,args.epochs); rows.append(row)
                print(f"seed={seed} method={method:9s} target={target:.2f} achieved={row['achieved_sparsity']:.3f} acc={row['accuracy']:.4f} active={row['active_weights']}/{row['total_weights']}")
    csv_path=os.path.join(args.outdir,'pareto_sweep.csv')
    with open(csv_path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Pareto plot: mean and population standard deviation across seeds.
    fig,ax=plt.subplots(figsize=(7,5))
    for method in ['magnitude','saliency']:
        xs=[]; means=[]; stds=[]
        for target in sparsities:
            subset=[r for r in rows if r['criterion']==method and abs(r['target_sparsity']-target)<1e-9]
            if not subset: continue
            xs.append(np.mean([r['achieved_sparsity'] for r in subset])); means.append(np.mean([r['accuracy'] for r in subset])); stds.append(np.std([r['accuracy'] for r in subset]))
        ax.errorbar(xs,means,yerr=stds,marker='o',capsize=3,label=method)
    ax.set_xlabel('Achieved weight sparsity'); ax.set_ylabel('Test accuracy'); ax.set_title('Sparsity–accuracy Pareto (mean ± std across seeds)'); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.plotdir,'pareto.png'),dpi=160); plt.close(fig)
    # Learning curve from a baseline run is produced separately by train.baseline.
if __name__=='__main__': main()
