import unittest
import numpy as np
from engine import Tensor, softmax_cross_entropy
from nn import MLP, Adam

class MaskedWeightTests(unittest.TestCase):
    def test_pruned_gradient_zero_and_state_reset_on_revival(self):
        model=MLP([2,3,2],seed=3); opt=Adam(model.parameters(),lr=.01)
        layer=model.layers[0]; w=layer.weight; maskmap={id(x.weight):x.mask for x in model.layers}
        # Seed nonzero optimizer history for the selected connection.
        opt.m[id(w)][0,0]=4.0; opt.v[id(w)][0,0]=9.0
        layer.mask[0,0]=False; w.data[0,0]=0.0
        changed=np.zeros_like(layer.mask); changed[0,0]=True; opt.on_mask_change(w,changed)
        x=np.array([[.3,-.2],[.1,.8]]); y=np.array([0,1])
        for p in model.parameters(): p.zero_grad()
        loss=softmax_cross_entropy(model(Tensor(x)),y); loss.backward()
        self.assertEqual(w.grad[0,0],0.0)
        opt.step(maskmap)
        self.assertEqual(w.data[0,0],0.0)
        # Revival starts with clean moments; no stale momentum can jump the weight.
        layer.mask[0,0]=True; opt.on_mask_change(w,changed)
        self.assertEqual(opt.m[id(w)][0,0],0.0); self.assertEqual(opt.v[id(w)][0,0],0.0)
        old=w.data[0,0]
        for p in model.parameters(): p.zero_grad()
        softmax_cross_entropy(model(Tensor(x)),y).backward(); opt.step(maskmap)
        self.assertNotEqual(w.data[0,0],old)

    def test_sparse_forward_matches_masked_dense_forward(self):
        model=MLP([2,5,3],seed=4); model.layers[0].mask[0,0]=False; model.layers[0].weight.data[0,0]=0
        model.layers[1].mask[1,1]=False; model.layers[1].weight.data[1,1]=0
        x=np.random.default_rng(5).normal(size=(7,2))
        dense=model(Tensor(x)).data; sparse=model.predict_sparse(x)
        np.testing.assert_allclose(sparse,dense,atol=1e-12,rtol=1e-12)

    
    def test_revived_coordinate_uses_fresh_adam_bias_correction(self):
        from engine import Tensor

        model = MLP([2, 3, 2], seed=10)
        opt = Adam(model.parameters(), lr=0.01)
        layer = model.layers[0]
        w = layer.weight
        key = id(w)

        # Simulate an optimizer that has already taken many steps.
        opt.t = 50

        # Revive one coordinate after resetting its optimizer state.
        changed = np.zeros_like(layer.mask, dtype=bool)
        changed[0, 0] = True
        layer.mask[0, 0] = True
        opt.on_mask_change(w, changed)

        # Give the revived coordinate a known gradient.
        for p in model.parameters():
            p.zero_grad()
        w.grad[0, 0] = 2.0

        before = w.data[0, 0]
        maskmap = {id(x.weight): x.mask for x in model.layers}
        opt.step(maskmap)

        # With a fresh coordinate, first-step Adam's update is approximately lr.
        actual_update = before - w.data[0, 0]
        self.assertAlmostEqual(actual_update, 0.01, places=6)
        self.assertEqual(opt.age[key][0, 0], 1)
    

if __name__=='__main__': unittest.main()
