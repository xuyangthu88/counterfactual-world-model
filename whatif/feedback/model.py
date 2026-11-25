"""Joint factual abduction and graph transport under a sustained feedback intervention."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import Mechanism, mlp, pairs, pairmask, tile, pool, H, U, FACTORS, WIDTH
from whatif.paired.fit import world

ATOMS = 17
TIMES = 4
STATE = FACTORS * WIDTH


class Facts(nn.Module):
    def __init__(self):
        super().__init__()
        initial = world()
        for key in ('prior', 'fact', 'latent', 'codes', 'revise', 'refine'):
            setattr(self, key, getattr(initial, key))

    def forward(self, d):
        # A persistent episode residual is conditional on the common observed probe.
        return Mechanism.abduct(self, dict(d, same=torch.ones_like(d['x'][:, 0, 0])))


class FeedbackWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.facts = Facts()
        self.context = mlp(9*3 + 3 + 12 + U, H, 64)
        self.factor = nn.ModuleList([mlp(H+3, WIDTH) for _ in range(FACTORS)])
        self.policy = mlp(11+7, STATE, 48)
        self.time = nn.Parameter(torch.randn(TIMES, 8) * .05)
        self.relation = mlp(2*STATE+6+8+1, 1+2*FACTORS**2, 72)
        self.step = nn.GRUCell(WIDTH, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.motion = nn.ModuleList([mlp(WIDTH, 3) for _ in range(FACTORS)])
        self.read = mlp(STATE + 18 + 8, 64, 80)
        self.scores = mlp(64, 4, 64)
        self.effect = mlp(2*64, 2*4, 64)
        self.events = mlp(3*64, 3*ATOMS, 96)
        self.register_buffer('grid', torch.linspace(-.75, .75, ATOMS))

    def forward(self, d):
        x, role, goal, a, w = [d[k] for k in ('x','role','goal','a','weights')]
        b, k, n = a.shape[:3]
        m, u, factual, kl = self.facts(d)
        obs = d['f1']-d['f0']
        ctx = self.context(torch.cat([x, x-goal[:,None], obs, role, m, u], -1))
        initial = torch.stack([self.factor[j](torch.cat([ctx,x[...,3*j:3*j+3]],-1))
                               for j in range(FACTORS)], -2)
        h = tile(initial,k)
        intervention = torch.cat([a,w[:,:,None,:]*role[:,None,:,1:2]],-1)
        source = (self.policy(intervention)-self.policy(torch.zeros_like(intervention))).reshape_as(h)
        h = h+source
        rel = torch.cat([x[...,:3],obs[...,:3]],-1)
        geo = tile(rel[:,None]-rel[:,:,None],k)
        prior = tile(factual,k)[...,None]
        gate_mask = tile(pairmask(role),k)
        mover = tile(role[...,0],k)
        gx = tile(goal,k)
        start = tile(x,k)
        futures, edges, scalars, reps = [], [], [], []
        for clock in self.time:
            features = torch.cat([pairs(h.flatten(-2)),geo,clock.expand(*geo.shape[:-1],-1),prior],-1)
            edge, fg, fw = self.relation(features).split([1,FACTORS**2,FACTORS**2],-1)
            logit = (edge+prior)[...,0]
            prob = logit.sigmoid()
            active = (prob>=.5).to(prob)+(prob-prob.detach())
            pf = fg.sigmoid()
            factors = (pf>=.5).to(pf)+(pf-pf.detach())
            weight = ((gate_mask*active)[...,None]*factors*fw.tanh()).unflatten(-1,(FACTORS,FACTORS))
            for _ in range(2):
                msg = torch.einsum('bijfg,bifh->bjgh',weight,h)
                h = self.norm(h + .5*self.step(msg.reshape(-1,WIDTH),h.reshape(-1,WIDTH)).reshape_as(h))
            delta = torch.cat([head(h[:,:,j]) for j,head in enumerate(self.motion)],-1)
            future = start+delta
            summary = torch.cat([pool(h.flatten(-2),mover),pool(future,mover)-gx,gx,clock.expand(b*k,-1)],-1)
            z = self.read(summary).reshape(b,k,-1)
            futures.append(future.reshape(b,k,n,9))
            edges.append(logit.reshape(b,k,n,n))
            scalars.append(self.scores(z))
            reps.append(z)
            h = h + .25*source
        return dict(y=torch.stack(futures,2),edge=torch.stack(edges,2),scalar=torch.stack(scalars,2),
                    z=torch.stack(reps,2),fact=factual,kl=kl)

    def contrasts(self, o, i, j, mover):
        z = o['z']
        l, r = z[:,i], z[:,j]
        diff = l-r
        def effect(z):
            return self.effect(torch.cat([z,z.abs()],-1))
        pos = pool(o['y'][...,:3],mover[:,None,None,:])
        base = torch.cat([pos,o['scalar'][...,3:]],-1)
        mu = base[:,i]-base[:,j]+.5*(effect(diff)[...,:4]-effect(-diff)[...,:4])
        scale = .012+F.softplus(effect(diff.abs())[...,4:])
        l,r = l[:,:,-1],r[:,:,-1]
        def event(a,b):
            return self.events(torch.cat([a,b,a-b],-1)).unflatten(-1,(3,ATOMS))
        logits = .5*(event(l,r)+event(r,l).flip((-1,-2)))
        p = logits.flatten(-2).softmax(-1).reshape_as(logits)
        return mu,scale,p


def preference(world, o, i, j, p):
    # Pairwise scores are antisymmetric; average against the same alternative policies.
    s = torch.arange(-1,2,device=p.device,dtype=p.dtype)[:,None]
    delta = s+world.grid
    mean = (p*delta).sum((-1,-2))
    direction = (p*delta.sign()).sum((-1,-2))
    law = mean+.25*direction
    win,lose = p[:,:,2].sum(-1),p[:,:,0].sum(-1)
    discord = win+lose
    q = (win/discord.clamp_min(1e-8)).clamp(1e-7,1-1e-7)
    kl = discord*(q-.5).sign()*(q*(2*q).log()+(1-q)*(2*(1-q)).log())
    v = o['scalar'][:,:,-1,:3].sigmoid()
    scores = dict(marginal=v[...,0]+.5*v[...,1]+.25*v[...,2])
    for key,value in [('mean',mean),('law',law),('kl',law+.5*kl)]:
        total = torch.zeros_like(scores['marginal'])
        total.index_add_(1,i,value)
        total.index_add_(1,j,-value)
        scores[key] = total/(total.size(1)-1)
    return scores
