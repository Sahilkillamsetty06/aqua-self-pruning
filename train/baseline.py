import argparse, json, os, sys
import numpy as np
from nn import MLP, Adam
from train.data import make_moons
from train.common import train_epoch,evaluate

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--out',default='results/baseline_history.json'); args=ap.parse_args()
    x,y,xt,yt=make_moons(seed=args.seed); model=MLP([2,32,32,2],seed=args.seed); opt=Adam(model.parameters(),lr=0.01); rng=np.random.default_rng(args.seed+100)
    hist=[]
    for ep in range(1,args.epochs+1):
        tr=train_epoch(model,opt,x,y,64,rng); vl,va=evaluate(model,xt,yt)
        row={'epoch':ep,'train_loss':tr,'test_loss':vl,'test_accuracy':va}; hist.append(row)
        if ep==1 or ep%10==0: print(f"epoch={ep:03d} train_loss={tr:.4f} test_loss={vl:.4f} test_acc={va:.4f}")
    os.makedirs(os.path.dirname(args.out) or '.',exist_ok=True)
    with open(args.out,'w') as f: json.dump({'seed':args.seed,'history':hist},f,indent=2)
if __name__=='__main__': main()
