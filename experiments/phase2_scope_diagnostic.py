"""
Quick diagnostic (about 3 minutes): which part of the model update separates the poisoned
client from honest ones on YOUR data? Run this before trusting a scope.

    python experiments/phase2_scope_diagnostic.py 0.2
    python experiments/phase2_scope_diagnostic.py 0.5

For each scope and round it prints the poisoned client's distance ratio against the
largest honest one ('sep' above 1.2 or so means a usable gap; about 1.0 means no signal),
and the cosine similarity of the poisoned client against the lowest honest one.
Scopes: all(18) = every trainable array, last_block(6), output(2) = output layer only.
"""
import sys, copy, os
sys.path.append("."); sys.path.append("experiments")
import numpy as np
from common_setup import *
from models.model import build_model
from fl.fl_client import FLClient
from fl.aggregation import fedavg
from abac.policy import default_phase1_clients
from anomaly.scoring import compute_delta, compute_round_features, get_trainable_mask, filter_by_mask

pf=float(sys.argv[1]); seed=0
ds=load_dataset_once(verbose=False); cw=compute_global_class_weight(ds["pool"])
cd,td,dim=build_clients(ds,seed=seed,poison_fraction=pf,verbose=False)
set_seed(seed); model=build_model(dim); mask=get_trainable_mask(model)
gw=model.get_weights()
attrs=default_phase1_clients()[:3]
clients=[FLClient(a.client_id,a,X,y,dim) for a,(X,y) in zip(attrs,cd[:3])]
scopes={"all(18)":None,"last_block(6)":6,"output(2)":2}
print("scope, round: poisoned dist_ratio / max honest dist_ratio | poisoned cos / min honest cos")
for rnd in range(1,5):
    tr={}
    for c in clients:
        w,n=c.train(gw,epochs=EPOCHS_PER_ROUND,batch_size=32,class_weight=cw)
        tr[c.client_id]=(w,n,filter_by_mask(compute_delta(w,gw),mask))
    for name,k in scopes.items():
        d={cid:(t[2] if k is None else t[2][-k:]) for cid,t in tr.items()}
        f=compute_round_features(d)
        hon=[v for c,v in f.items() if c!="client_2"]
        p=f["client_2"]
        print(f"{name:14s} r{rnd}: dist {p['dist_ratio']:.2f} / {max(h['dist_ratio'] for h in hon):.2f} "
              f"(sep {p['dist_ratio']/max(h['dist_ratio'] for h in hon):.2f}) | cos {p['cos_sim']:+.2f} / {min(h['cos_sim'] for h in hon):+.2f}")
    gw=fedavg([t[0] for t in tr.values()],[t[1] for t in tr.values()])
