"""Shared state adapters for the three finite-budget baseline comparisons."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import mlp,pool,tile,pairs,labeled
from .native import A,D


def node_inputs(d):
    k = d['a'].size(1)
    history = torch.cat([d['f0'],d['f1']-d['f0'],d['fa']],-1)
    return torch.cat([tile(d['x'],k), tile(d['role'],k), tile(history,k),
                      d['a'].flatten(0,1)],-1)


class CoPhy(nn.Module):
    def __init__(self):
        super().__init__()
        from whatif.crossbench.models import cophy_class
        self.core = cophy_class()()
        self.project = mlp(D+A+3,3)
        self.decode = mlp(3, D)
        self.reward = mlp(2*D+3,3)

    def tokens(self,x,a,r):
        return x[...,:3]+self.project(torch.cat([x,a,r],-1))

    def forward(self,d):
        b,k,n,_ = d['a'].shape
        role = d['role']
        present = role.sum(-1)
        ab = torch.stack([self.tokens(d[t],d['fa'],role) for t in ('f0','f1')],1)
        ctx = self.core.rnn_on_AB_up(self.core.gcn_on_AB(ab,present))
        c = self.tokens(tile(d['x'],k),d['a'].flatten(0,1),tile(role,k))
        pred,stable = self.core.pred_D(tile(ctx,k),c,tile(present,k),T=1)
        future = tile(d['x'],k)+self.decode(pred[:,0]-c)+F.pad(pred[:,0]-c,(0,6))
        movable = tile(role[...,0],k)
        g = tile(d['goal'],k)
        r = self.reward(torch.cat([pool(future,movable)-g,g,pool(pred[:,0],movable)],-1))
        return dict(y=future.reshape(b,k,n,D),r=r.reshape(b,k,3),
                    stable=stable[:,0,:,0].reshape(b,k,n))


class PETS(nn.Module):
    def __init__(self,device):
        super().__init__()
        from mbrl.models import GaussianMLP
        inp = 3*D+2*A+3
        self.emb = mlp(inp,32)
        self.net = GaussianMLP(inp+32+D,D+3,device,num_layers=3,ensemble_size=5,hid_size=96)

    def forward(self,d):
        b,k,n,_ = d['a'].shape
        node = node_inputs(d)
        shared = pool(self.emb(node),tile(d['role'].sum(-1),k))
        context = torch.cat([shared,tile(d['goal'],k)],-1)[:,None].expand(-1,n,-1)
        z = torch.cat([node,context],-1).flatten(0,1)
        mean,lv = self.net(z,use_propagation=False)
        mean = mean.reshape(5,b,k,n,D+3)
        lv = lv.reshape_as(mean)
        y = d['x'][:,None]+mean[...,:D].mean(0)
        r = pool(mean[...,D:].mean(0),d['role'][:,None,:,0])
        return dict(y=y,r=r,ensemble=mean,lv=lv)


class NCII(nn.Module):
    def __init__(self):
        super().__init__()
        width = 3*D+2*A+3
        self.point = mlp(2*width,32)
        self.transition = mlp(width+32,D*2)
        self.interaction = mlp(2*width,1)

    def forward(self,d):
        node = node_inputs(d)
        b,k,n,_ = d['a'].shape
        # First pair axis is source; max-pool incoming PointNet features.
        pair = pairs(node)
        points = self.point(pair)
        valid = tile(d['role'].sum(-1),k).bool()
        valid = valid[:,:,None] & valid[:,None,:] & ~torch.eye(n,device=node.device,dtype=torch.bool)
        def predict(mask):
            pooled = points.masked_fill(~mask[...,None],-1e4).amax(1)
            pooled = torch.where(mask.any(1)[...,None],pooled,0.)
            mu,lv = self.transition(torch.cat([node,pooled],-1)).chunk(2,-1)
            return mu,lv.clamp(-4,2)
        full,logsd = predict(valid)
        result = dict(edge=self.interaction(pair).reshape(b,k,n,n),y=d['x'][:,None]+full.reshape(b,k,n,D))
        if self.training:
            mask = valid & (torch.rand_like(valid.float())>.25)
            subset,sd = predict(mask)
            target = (d['y']-d['x'][:,None]).flatten(0,1)
            nll = lambda m,s: (.5*((target-m)*(-s).exp()).square()+s).mean(-1)
            with torch.no_grad():
                full_loss = nll(full,logsd)
                gaps = []
                for i in range(n):
                    keep = valid.clone()
                    keep[:,i] = False
                    gaps.append(nll(*predict(keep))-full_loss)
                pseudo = (torch.stack(gaps,1)>.10).float().reshape(b,k,n,n)
            mask_edges = labeled(d['role'])[:,None].expand_as(pseudo)
            bce = F.binary_cross_entropy_with_logits(result['edge'],pseudo,reduction='none')
            result['loss'] = (pool((nll(full,logsd)+nll(subset,sd))[...,None],tile(d['role'].sum(-1),k)).mean()*.5
                              +(bce*mask_edges).sum()/mask_edges.sum().clamp_min(1))
        return result

