"""Roll two actions or an edge intervention through the same inferred physical world."""
import torch
from torch import nn
from torch.nn import functional as F

from whatif.mechanism.model import K, U, mlp, pairmask, pairs, pool, tile
from .data import clock, pair_ids

FACTORS = 3
WIDTH = 20
FEATURE = 64
COMPONENTS = 4


def relative(position):
    return position.unsqueeze(-3) - position.unsqueeze(-2)


def average(x, weight):
    weight = weight.expand_as(x)
    return (x*weight).sum()/weight.sum().clamp_min(1)


def hard_gate(p, soft):
    discrete = (p >= .5).to(p) + (p-p.detach())
    return soft*p + (1-soft)*discrete


class Facts(nn.Module):
    """Estimate a physical context and correct a factual graph using observed innovations."""
    def __init__(self):
        super().__init__()
        self.local = mlp(9+7+3+2, FEATURE, FEATURE)
        self.prior = mlp(2*FEATURE+3, 1, FEATURE)
        self.motion = mlp(2*FEATURE, 9, FEATURE)
        self.history = mlp(FEATURE+18, FEATURE, FEATURE)
        self.attend = nn.Linear(FEATURE, 1)
        self.latent = mlp(FEATURE+1, K+2*U, FEATURE)
        self.mechanisms = nn.Parameter(torch.randn(K, 12)*.2)
        self.correct = mlp(2*(FEATURE+12+U+9)+1, 1, FEATURE)
        self.refine = mlp(FEATURE+2, K+2*U, FEATURE)

    def forward(self, d):
        role, factual = d['role'], d['fact']
        current, next_pose = factual[:, :-1], factual[:, 1:]
        t = current.size(1)
        command = d['fa'][:, None].expand(-1, t, -1, -1)
        local = self.local(torch.cat([current, command, role[:, None].expand(-1,t,-1,-1),
                                      clock(d['widths'])[:, :, None].expand(-1,-1,role.size(1),-1)], -1))
        prior = self.prior(torch.cat([pairs(local), relative(current[..., :3])], -1))[..., 0]
        valid = pairmask(role)
        prob = prior.sigmoid()*valid[:, None]
        message = torch.einsum('btij,btih->btjh', prob, local)
        motion = self.motion(torch.cat([local, message/prob.sum(-2).clamp_min(1)[...,None]], -1))
        innovation = next_pose-current-motion
        enc = self.history(torch.cat([local, motion, innovation], -1))
        alpha = self.attend(enc).softmax(1)
        history = (enc*alpha).sum(1)
        innovation_avg = (innovation*alpha).sum(1)
        alpha = alpha[...,0]
        initial = (prior*(alpha[...,None]+alpha[:,:,None,:])*.5).sum(1)

        def degree(logit):
            return (logit.sigmoid()*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)

        def posterior(v):
            coef, mean, log_sd = v.split([K,U,U], -1)
            return coef.softmax(-1)@self.mechanisms, mean, log_sd.clamp(-4, 1)

        first = self.latent(torch.cat([history, degree(initial)], -1))
        m, u, _ = posterior(first)
        edit = torch.cat([pairs(torch.cat([history,m,u,innovation_avg], -1)), initial[...,None]], -1)
        factual_graph = initial + self.correct(edit)[...,0]
        m, mean, scale = posterior(first + self.refine(torch.cat([history, degree(initial), degree(factual_graph)], -1)))
        eps = d.get('eps')
        if eps is None:
            eps = torch.randn_like(mean) if self.training else torch.zeros_like(mean)
        kl = .5*(mean.square()+(2*scale).exp()-1-2*scale)
        return dict(m=m, u=mean+eps*scale.exp(), graph=factual_graph,
                    kl=average(kl, role.sum(-1)[...,None]), innovation=innovation)


class World(nn.Module):
    def __init__(self):
        super().__init__()
        self.facts = Facts()
        self.node = mlp(9+9+3+12+U, FEATURE, FEATURE)
        self.factor = nn.ModuleList([mlp(FEATURE+3, WIDTH, FEATURE) for _ in range(FACTORS)])
        self.action = mlp(7+2, FACTORS*WIDTH, FEATURE)
        self.endpoint = mlp(2*FACTORS*WIDTH+9+9, FEATURE, FEATURE)
        self.edge = mlp(2*FEATURE+3+1+2, 4+2*FACTORS**2, 96)
        self.message = nn.Linear(WIDTH, WIDTH, bias=False)
        self.recurrent = nn.GRUCell(WIDTH+2, WIDTH)
        self.delta = nn.ModuleList([mlp(WIDTH+3, 3, FEATURE) for _ in range(FACTORS)])
        self.reward = mlp(FACTORS*WIDTH+18, FEATURE, FEATURE)
        self.reward_mean = nn.Linear(FEATURE, 1)
        self.regime = mlp(FEATURE, COMPONENTS)
        self.position_dist = mlp(WIDTH, 3*3*COMPONENTS, FEATURE)
        self.reward_dist = nn.Linear(FEATURE, 3*COMPONENTS)

    def interaction(self, end, position, prior, dt, mask, omit, carry, soft):
        n = position.size(1)
        features = torch.cat([pairs(end), relative(position[..., :3]), prior[...,None],
                              dt[:,None,None].expand(-1,n,n,-1)], -1)
        p, c, strength, persist, opening, value = self.edge(features).split([1,1,1,1,9,9], -1)
        logit = p[...,0] + prior
        logit = .5*(logit + logit.transpose(-1,-2))
        contact = logit.sigmoid()
        conduct = (c[...,0]+1.5).sigmoid()
        magnitude = F.softplus(strength[...,0])
        carry = 1-(1-contact*conduct)*(1-carry*persist[...,0].sigmoid())
        edge_gate = mask*omit*hard_gate(contact, soft)*conduct*(1+magnitude.tanh())
        factor_gate = hard_gate(opening.sigmoid(), soft)*value.tanh()
        weights = edge_gate[...,None,None]*factor_gate.unflatten(-1,(FACTORS,FACTORS))
        return weights, logit, carry, magnitude

    def rollout(self, d, node, facts, action, reset, keep=None, soft=0.):
        x, role = d['x'], d['role']
        b, k, n = action.shape[:3]
        position = tile(x, k) + reset.flatten(0, 1)
        start = position
        ctx = tile(node, k)
        mask = tile(pairmask(role), k)
        prior = tile(facts['graph'], k)
        local_action = action.flatten(0, 1)
        omit = torch.ones_like(mask) if keep is None else keep.flatten(0, 1)
        state = torch.stack([f(torch.cat([ctx, position[...,i:i+3]], -1))
                             for i, f in zip((0,3,6), self.factor)], -2)
        message = torch.zeros_like(state)
        last = torch.zeros_like(position)
        carry = torch.zeros_like(prior)
        outputs = []
        for dt in clock(d['widths']).unbind(1):
            dt = tile(dt, k)
            impulse = self.action(torch.cat([local_action, dt[:,None].expand(-1,n,-1)], -1))
            state = state + local_action[...,0,None,None]*impulse.unflatten(-1, (FACTORS,WIDTH))
            end = self.endpoint(torch.cat([state.flatten(-2), message.flatten(-2), position, last], -1))
            weights, logit, carry, magnitude = self.interaction(end, position, prior, dt, mask, omit, carry, soft)
            message = torch.einsum('bijfg,bifh->bjgh', weights, self.message(state))
            time = dt[:, None, None].expand(-1, n, FACTORS, -1)
            state = state + self.recurrent(torch.cat([message, time], -1).flatten(0,2),
                                           state.flatten(0,2)).reshape_as(state)
            motion = torch.cat([f(torch.cat([state[:,:,i], start[...,3*i:3*i+3]], -1))
                                for i,f in enumerate(self.delta)], -1)
            new_position = start+motion
            last = new_position-position
            position = new_position
            outputs.append((position, logit, torch.logit(carry.clamp(1.e-5, 1-1.e-5)), magnitude, state))
        return [torch.stack(v, 1).unflatten(0, (b,k)) for v in zip(*outputs)]

    def forward(self, d, removed=None, *, soft=0.):
        facts = self.facts(d)
        x, role, action = d['x'], d['role'], d['a']
        node = self.node(torch.cat([x, d['fact'][:,-1]-d['fact'][:,0], role, facts['m'], facts['u']], -1))
        ref = d['fa'][:,None]
        # This reference is used only when a factual continuation from x was observed.
        action = torch.cat([action,ref], 1)
        reset = torch.cat([d['reset'], torch.zeros_like(d['reset'][:, :1])], 1)
        keep = None
        b, k, n = d['a'].shape[:3]
        if removed is not None:
            variants = removed.size(2)
            i,j = pair_ids(n, x.device)
            delete = x.new_zeros(b,k,variants,n,n)
            ix = i[removed][...,None,None].expand(-1,-1,-1,1,n)
            # A single indicator plane for each unordered intervention.
            delete.scatter_(-2, ix, F.one_hot(j[removed], n)[...,None,:].to(x))
            delete = delete + delete.transpose(-1,-2)
            keep = torch.cat([x.new_ones(b,k+1,n,n), 1-delete.flatten(1,2)], 1)
            action = torch.cat([action, d['a'].repeat_interleave(variants, dim=1)], 1)
            reset = torch.cat([reset, d['reset'].repeat_interleave(variants, dim=1)], 1)
        position, edges, causal, strength, state = self.rollout(d, node, facts, action, reset, keep, soft)
        regular = position[:, :k]
        final = regular[:,:,-1]+d['same'][:,None,None,None]*(d['fact'][:,None,-1]-position[:,k:k+1,-1])
        pose = torch.cat([regular[:,:,:-1], final[:,:,None]], 2)
        state = state[:,:k]
        movable = role[:, None, None, :, 0]
        goal = d['goal'][:,None,None].expand(-1,k,pose.size(2),-1)
        reward_features = self.reward(torch.cat([pool(state.flatten(-2),movable),
                                                pool(pose,movable)-goal, goal], -1))
        w = self.regime(pool(node, role.sum(-1))).softmax(-1)

        def distribution(raw):
            offset, scale, load = raw.unflatten(-1, (COMPONENTS,3)).unbind(-1)
            weight = w.reshape(b, *([1]*(offset.ndim-2)), COMPONENTS)
            return offset-(offset*weight).sum(-1,keepdim=True), .025+F.softplus(scale), load

        pose_dist = distribution(self.position_dist(state).unflatten(-1,(3,3*COMPONENTS)).flatten(-3,-2))
        reward_dist = distribution(self.reward_dist(reward_features))
        out = dict(pose=pose, edge=edges[:,:k], causal=causal[:,:k], strength=strength[:,:k],
                   reward=self.reward_mean(reward_features)[...,0], w=w, p_dist=pose_dist, r_dist=reward_dist,
                   **facts)
        if removed is not None:
            off = position[:, k+1:].unflatten(1, (k,variants))
            out['edge_effect'] = regular[:,:,None,...,:3]-off[...,:3]
        return out


def contrast(mean, dist, w, i, j, independent=False):
    loc, sd, load = dist
    loc = mean[...,None]+loc
    if independent:
        var = sd.square()+load.square()
        m = loc[:,i].unsqueeze(-1)-loc[:,j].unsqueeze(-2)
        sd = (var[:,i].unsqueeze(-1)+var[:,j].unsqueeze(-2)).sqrt()
        return m.flatten(-2), sd.flatten(-2), (w[:,:,None]*w[:,None,:]).flatten(1)
    var = sd[:,i].square()+sd[:,j].square()+(load[:,i]-load[:,j]).square()
    return loc[:,i]-loc[:,j], var.sqrt(), w


def build_world(name):
    from .collect import MOTION_PROTOCOL
    spec = name.removesuffix('-calibrated')
    if spec in MOTION_PROTOCOL['architecture']:
        from .motion import MotionWorld
        return MotionWorld(MOTION_PROTOCOL['architecture'][spec])
    return World()
