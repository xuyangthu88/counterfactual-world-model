"""Published CoPhyNet and MBRL-Lib PETS, on the same open-loop records."""
import argparse
import ast
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from whatif.crossbench.models import cophy_class
from .collect import DOMAINS, OUT, PROTOCOL, ROOT
from .data import clock, load, subset

CO_PHY_SOURCE = ROOT/'work/stage7/upstream/cophy/cf_learning/main.py'


def cophy_loss():
    nodes = ast.parse(CO_PHY_SOURCE.read_text())
    original = next(n for n in nodes.body if getattr(n,'name',None)=='get_losses')
    space = {'torch':torch, 'F':F}
    exec(compile(ast.Module([original],type_ignores=[]), str(CO_PHY_SOURCE), 'exec'), space)
    return space['get_losses']


def cophy_inputs(d):
    b,k,n = d['a'].shape[:3]
    present = d['role'].sum(-1)
    fact = d['fact'][..., :3]
    initial = (d['x'][:,None]+d['reset'])[..., :3]
    truth = d['pose'][..., :3]
    if 'off' not in d:
        # The released model expects 3D points in AB and C. The two visible
        # waypoint targets are additional static points in these sequences.
        def points(x,a):
            return x[...,None,:] + a[...,1:].unflatten(-1, (2,3))

        fact_points = (points(fact[:,0],d['fa'])*d['fa'][...,0,None,None]).sum(-3)
        future_points = (points(d['x'][:,None,:,:3],d['a'])*d['a'][...,0,None,None]).sum(-3)
        fact = torch.cat([fact,fact_points[:,None].expand(-1,fact.size(1),-1,-1)],-2)
        initial = torch.cat([initial,future_points],-2)
        truth = torch.cat([truth, future_points[:,:,None].expand(-1,-1,truth.size(2),-1,-1)],-2)
        present = F.pad(present, (0,2), value=1)
    rep = lambda x: x[:,None].expand(-1,k,*x.shape[1:]).flatten(0,1)
    return rep(fact), rep(present), initial.flatten(0,1), truth.flatten(0,1), (b,k,n)


def cophy_predict(model,d):
    fact, present, initial, truth, (b,k,n) = cophy_inputs(d)
    feat = model.gcn_on_AB(fact,present)
    m = model.rnn_on_AB_up(feat)
    pred, logits = model.pred_D(m, initial, present, T=truth.size(1))
    return dict(raw=pred, stability=logits[...,0], mask=present, truth=truth, initial=initial,
                pose=pred[:,:,:n].unflatten(0,(b,k)))


def train_cophy(domain):
    device='cuda'
    train, val = [load(domain,split,device) for split in ('train','val')]
    model = cophy_class()().to(device)
    original = cophy_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history=[]
    start = time.monotonic()
    # Upstream script specifies --epochs 50; its loop is range(1, epochs).
    for epoch in range(1, PROTOCOL['budgets']['cophynet_epochs']):
        model.train()
        # 32 single-action sequences per batch. Different siblings are not
        # contrasted in the loss, and share a shuffled example population.
        order = torch.randperm(train['a'].shape[0]*train['a'].shape[1], device=device)
        for ix in order.split(32):
            rows, cols = ix//train['a'].shape[1], ix%train['a'].shape[1]
            d = subset(train, rows)
            for key in ('a','reset','pose','edge','reward','off'):
                if key in d:
                    d[key] = d[key][torch.arange(len(ix),device=device),cols][:,None]
            o = cophy_predict(model,d)
            seq = torch.cat([o['initial'][:,None],o['truth']],1)
            indices = torch.arange(o['truth'].size(1),device=device)
            stable = ((seq[:,(indices-2).clamp_min(0)]-seq[:,(indices+2).clamp_max(seq.size(1)-1)])
                      .abs().sum(-1)>.05).float()*o['mask'][:,None]
            loss, _ = original(o['raw'],o['stability'],o['mask'],o['truth'],stable,o['mask'],
                               w_stab=1.,w_pose=10.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch%5==0 or epoch==PROTOCOL['budgets']['cophynet_epochs']-1:
            model.eval()
            with torch.no_grad():
                mse=[]
                for ix in torch.arange(len(val['x']),device=device).split(32):
                    d=subset(val,ix)
                    err=cophy_predict(model,d)['pose']-d['pose'][...,:3]
                    mse.append(float(((err.square().sum(-1)*d['role'][:,None,None,:,0]).sum()
                                      / (d['role'][...,0].sum()*err.size(1)*err.size(2))).sqrt()))
            history.append(dict(epoch=epoch, validation_pose_batch_rmse=float(np.mean(mse)),
                                seconds=round(time.monotonic()-start,1)))
            print(domain,'cophynet',history[-1],flush=True)
            (OUT/f'cophynet-{domain}-fit.json').write_text(json.dumps(history,indent=2)+'\n')
    torch.save(model.state_dict(),OUT/f'cophynet-{domain}.pt')


def pets_context(d):
    b,k,n = d['a'].shape[:3]
    fixed = torch.cat([d['fact'].flatten(1), d['fa'].flatten(1), d['role'].flatten(1),
                       d['x'].flatten(1),d['goal']], -1)
    cmd = torch.cat([fixed[:,None].expand(-1,k,-1),d['a'].flatten(2)],-1)
    time = clock(d['widths'])[:,None].expand(-1,k,-1,-1)
    return torch.cat([cmd[:,:,None].expand(-1,-1,time.size(2),-1),time],-1)


def pets_batch(d):
    from mbrl.types import TransitionBatch
    pose = d['pose'].flatten(-2)
    initial = (d['x'][:,None]+d['reset']).flatten(-2)
    obs = torch.cat([initial[:,:,None],pose[:,:,:-1]],2)
    kwargs = tuple(v.flatten(0,2).cpu().numpy() for v in (obs,pets_context(d),pose,d['reward']))
    empty = np.zeros(len(kwargs[0]),dtype=bool)
    return TransitionBatch(*kwargs, empty, empty)


def pets_build(in_size, out_size, device, learned_rewards):
    from mbrl.models import GaussianMLP, OneDTransitionRewardModel
    model = GaussianMLP(in_size,out_size,device, num_layers=4,ensemble_size=7,hid_size=200,
                        propagation_method='fixed_model',
                        activation_fn_cfg={'_target_':'torch.nn.SiLU'})
    return OneDTransitionRewardModel(model,normalize=True,normalize_double_precision=True,
                                    num_elites=5,learned_rewards=learned_rewards)


def train_pets(domain):
    from mbrl.models import ModelTrainer
    from mbrl.util.replay_buffer import BootstrapIterator
    device='cuda'
    data=pets_batch(load(domain,'train'))
    rewards=domain!='cophy'
    out_size=data.obs.shape[-1]+int(rewards)
    in_size=data.obs.shape[-1]+data.act.shape[-1]
    model=pets_build(in_size,out_size,device,rewards)
    model.update_normalizer(data)
    batches=BootstrapIterator(data,32,7,shuffle_each_epoch=True,permute_indices=True,
                              rng=np.random.default_rng(PROTOCOL['seed']))
    trainer=ModelTrainer(model,optim_lr=1.e-5,weight_decay=5.e-4)
    log=[]
    start=time.monotonic()

    def record(_model, _iteration, epoch, loss, score, best):
        log.append(dict(epoch=epoch+1, nll=float(loss), mse=float(score.mean()), seconds=round(time.monotonic()-start,1)))
        print(domain,'pets',log[-1],flush=True)
        (OUT/f'pets-{domain}-fit.json').write_text(json.dumps(log,indent=2)+'\n')

    trainer.train(batches, num_epochs=PROTOCOL['budgets']['pets_epochs'],patience=25,callback=record,silent=True)
    norm=model.input_normalizer
    torch.save(dict(model=model.state_dict(), mean=norm.mean, std=norm.std,
                    elites=[int(i) for i in model.model.elite_models],
                    in_size=in_size, out_size=out_size, learned_rewards=rewards),OUT/f'pets-{domain}.pt')


def pets_restore(domain,device):
    ck=torch.load(OUT/f'pets-{domain}.pt',map_location=device)
    model=pets_build(ck['in_size'],ck['out_size'],device,ck['learned_rewards'])
    model.load_state_dict(ck['model'])
    model.input_normalizer.mean=ck['mean']
    model.input_normalizer.std=ck['std']
    model.model.set_elite(ck['elites'])
    return model.eval()


def pets_predict(model, d, rng):
    b,k,n = d['a'].shape[:3]
    particles = 20
    initial = (d['x'][:,None]+d['reset']).flatten(-2)
    position = initial[:,:,None].expand(-1,-1,particles,-1).reshape(-1,n*9)
    state = model.reset(position,rng=rng)
    results=[]
    for cmd in pets_context(d).unbind(2):
        cmd=cmd[:,:,None].expand(-1,-1,particles,-1).flatten(0,2)
        pose,reward,_,state=model.sample(cmd,state,deterministic=False,rng=rng)
        shape=(b,k,particles,n,9)
        results.append((pose.reshape(shape), None if reward is None else reward.reshape(b,k,particles)))
    samples=torch.stack([point for point,_ in results],3)
    reward=torch.stack([r for _,r in results],-1) if model.learned_rewards else None
    return dict(pose=samples.mean(2), draws=samples, reward=reward.mean(2) if reward is not None else None,
                reward_draws=reward)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('name', choices=('cophynet','pets'))
    p.add_argument('domain', choices=DOMAINS)
    args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(PROTOCOL['seed'])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    (train_cophy if args.name=='cophynet' else train_pets)(args.domain)
