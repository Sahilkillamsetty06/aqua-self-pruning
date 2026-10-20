import numpy as np

class GlobalPruner:
    def __init__(self, model, optimizer): self.model=model; self.optimizer=optimizer
    def prune_to_sparsity(self, target_sparsity, criterion='saliency', gradients=None):
        """Prune exactly the required global fraction of weight connections.

        Saliency uses |w * dL/dw|, a first-order Taylor estimate of the loss
        change caused by setting a currently active weight to zero.
        """
        weights=self.model.weights(); masks=self.model.masks()
        total=sum(m.size for m in masks)
        desired=int(round(total*(1-target_sparsity)))
        active=sum(int(m.sum()) for m in masks)
        remove=max(0, active-desired)
        if remove == 0: return
        scores=[]; coords=[]
        for li,(w,m) in enumerate(zip(weights,masks)):
            if criterion == 'magnitude': score=np.abs(w.data)
            elif criterion == 'saliency':
                if gradients is None: grad=w.grad
                else: grad=gradients[li]
                score=np.abs(w.data*grad)
            else: raise ValueError('criterion must be magnitude or saliency')
            idx=np.argwhere(m)
            scores.extend(score[m].tolist())
            coords.extend((li,int(i),int(j)) for i,j in idx)
        order=np.argsort(np.asarray(scores), kind='stable')[:remove]
        for k in order:
            li,i,j=coords[int(k)]; layer=self.model.layers[li]
            layer.mask[i,j]=False; layer.weight.data[i,j]=0.0
            changed=np.zeros_like(layer.mask); changed[i,j]=True
            self.optimizer.on_mask_change(layer.weight, changed)
