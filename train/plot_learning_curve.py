import argparse,json,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='results/baseline_history.json'); ap.add_argument('--output',default='plots/baseline_learning_curve.png'); args=ap.parse_args()
    with open(args.input) as f: data=json.load(f)
    h=data['history']; epochs=[r['epoch'] for r in h]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(epochs,[r['train_loss'] for r in h],label='train'); axes[0].plot(epochs,[r['test_loss'] for r in h],label='test'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-entropy loss'); axes[0].set_title('Learning curve: loss'); axes[0].grid(True,alpha=.3); axes[0].legend()
    axes[1].plot(epochs,[r['test_accuracy'] for r in h],label='test accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0,1.02); axes[1].set_title('Learning curve: accuracy'); axes[1].grid(True,alpha=.3); axes[1].legend()
    fig.tight_layout(); os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True); fig.savefig(args.output,dpi=160); plt.close(fig)
if __name__=='__main__': main()
