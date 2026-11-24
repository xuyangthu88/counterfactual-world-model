"""Action-conditioned transfer of pose residuals over a variable-size set of objects."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LENGTH = .065


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,outputs))


class ContactTransfer(nn.Module):
    """One parameter set for every hand edge, and one for every ordered block pair."""
    def __init__(self, gated):
        super().__init__()
        self.gated = gated
        self.hand = mlp(15,4)
        self.interaction = mlp(14,7)
        self.coast = nn.Parameter(torch.tensor([-4.,-4.,-4.]))

    def forward(self, state, mask, tip, command, physics, noise):
        # All leading dimensions may include scene, candidate and particle axes.
        # Command gives the start in the push frame and the commanded travel.
        n = state.shape[-2]
        px,py,yaw,vx,vy,omega = state.unbind(-1)
        def per_node(x):
            return x.unsqueeze(-1).expand_as(px)
        stroke = (command[...,2]+noise)/LENGTH
        logm = (physics[...,0]/.075).log()
        logf = (physics[...,1]/.5).log()
        dx = (per_node(command[...,0])-px)/LENGTH
        dy = (per_node(command[...,1])-py)/LENGTH
        variables = [px/LENGTH,py/LENGTH,yaw.sin(),yaw.cos(),vx,vy,omega/10,
                     dx,dy,per_node(stroke),(per_node(tip[...,0])-px)/LENGTH,
                     (per_node(tip[...,1])-py)/LENGTH,per_node(tip[...,2])/.1,
                     per_node(logm),per_node(logf)]
        h = self.hand(torch.stack(variables,dim=-1))
        # Wide envelope of a finger sweeping in the -x direction.
        hand_possible = mask & (dx < per_node(stroke)+1.3) & (dy.abs()<1.75)
        hand_gate = h[...,0].sigmoid() if self.gated else torch.ones_like(px)
        velocity = torch.stack((vx/LENGTH,vy/LENGTH,omega),dim=-1)
        direct = velocity*self.coast.sigmoid() + (hand_gate*hand_possible)[...,None]*h[...,1:]
        direct = direct*mask[...,None]
        # Pair dimensions are sender, receiver. Edges follow the commanded push axis.
        rx = (px.unsqueeze(-2)-px.unsqueeze(-1))/LENGTH
        ry = (py.unsqueeze(-2)-py.unsqueeze(-1))/LENGTH
        valid = mask.unsqueeze(-2) & mask.unsqueeze(-1) & (rx < 0)
        extent = .5*(yaw.cos().abs()+yaw.sin().abs())
        gap = (-rx-extent.unsqueeze(-1)-extent.unsqueeze(-2)).clamp_min(0)
        def send(x):
            return x.unsqueeze(-1).expand_as(rx)
        def receive(x):
            return x.unsqueeze(-2).expand_as(rx)
        motion = direct
        pair_logits = torch.zeros_like(rx)
        reaction = torch.zeros_like(direct)
        for _ in range(n-1):
            travel = send((-motion[...,0]).clamp_min(0))
            drive = torch.minimum(travel,F.softplus(travel-gap,beta=12))
            feat = [rx,ry,gap,drive,send(yaw.sin()),send(yaw.cos()),receive(yaw.sin()),
                    receive(yaw.cos()),send(motion[...,0]),send(motion[...,1]),send(motion[...,2]),
                    receive(direct[...,0]),send(per_node(logm)),send(per_node(logf))]
            coeff = self.interaction(torch.stack(feat,dim=-1))
            pair_logits = 8*(travel-gap) + coeff[...,0]
            gate = pair_logits.sigmoid() if self.gated else torch.ones_like(rx)
            strength = (drive*gate*valid)[...,None]
            passed = strength*coeff[...,1:4]
            reaction = (strength*coeff[...,4:]).sum(dim=-2)
            motion = direct + passed.sum(dim=-3)
        return (motion+reaction)*mask[...,None], h[...,0],pair_logits,valid,hand_possible


class Branches:
    def __init__(self, filename, device):
        f = np.load(filename)
        self.n = int(f["count"].max())
        self.ids=f["seed"]
        def tensor(x):
            return torch.as_tensor(x,dtype=torch.float32,device=device)
        count=f["count"]
        self.mask=tensor(np.arange(self.n)[None,:]<count[:,None]).bool()
        states = f["states"]
        tips = f["tips"]
        goal = f["goal"]
        frame = float(f["frame"]) if "frame" in f else 0.
        if frame:
            # Row vectors in world coordinates rotate back to the commanded push axis.
            c, s = np.cos(frame), np.sin(frame)
            rotation = np.array([[c, -s], [s, c]])
            states[..., :2] = states[..., :2] @ rotation
            states[..., 3:5] = states[..., 3:5] @ rotation
            yaw = states[..., 2] - frame
            states[..., 2] = np.arctan2(np.sin(yaw), np.cos(yaw))
            tips[..., :2] = tips[..., :2] @ rotation
            goal = goal @ rotation
        self.state=tensor(states[:,:,0,0,:self.n,:])
        self.end=tensor(states[:,:,:,-1,:self.n,:3])
        edge=f["edges"].any(axis=3)
        self.hand=tensor(edge[:,:,:,0,:self.n])
        self.pair=tensor(edge[:,:,:,1:self.n+1,:self.n])
        self.physics=tensor(f["physics"])
        self.noise=tensor(f["noise"])
        self.tip=tensor(tips)
        self.actions=tensor(f["actions"])
        self.goal=tensor(goal)
        self.probe=int(f["probe"])
        self.target=torch.as_tensor(count-1,device=device)
        self.scale=tensor([LENGTH,LENGTH,1.])
        self.delta=(self.end-self.state[:,:,None,:,:3])/self.scale
        self.delta[...,2]=wrap(self.delta[...,2])

    def inputs(self, ids, time, actions, phys, noise):
        state=self.state[ids,time]
        # Translate the source-targeted command into the common push frame.
        # The learned dynamics see its coordinates without a source object ID.
        source=state[...,0,:2]
        cmd=torch.stack((source[...,0]+.065,source[...,1]+actions[...,0],actions[...,1]),dim=-1)
        return state,self.mask[ids],self.tip[ids,time],cmd,phys,noise


def wrap(a):
    return torch.atan2(a.sin(),a.cos())


def pose_loss(pred, target, mask):
    err=pred-target
    err=torch.cat((err[...,:2],wrap(err[...,2])[...,None]),dim=-1)
    values=F.smooth_l1_loss(err,torch.zeros_like(err),beta=.12,reduction="none")
    return (values*mask[...,None]).sum()/(3*mask.sum()).clamp_min(1)


def objective(model, data, ids, horizon, order=None):
    # The action index runs across sibling branches for each snapshot.
    a=torch.arange(len(data.actions),device=data.state.device)[None,:].expand(len(ids),-1)
    ids=ids[:,None].expand_as(a)
    horizon=horizon[:,None].expand_as(a)
    args=data.inputs(ids,horizon,data.actions[a],data.physics[ids],data.noise[ids,horizon])
    pred,hand,pair,valid,_=model(*args)
    target=data.delta[ids,horizon,a]
    mask=data.mask[ids]
    order=torch.arange(a.shape[1],device=a.device).roll(1) if order is None else order
    loss=pose_loss(pred,target,mask)+pose_loss(pred-pred[:,order],target-target[:,order],mask)
    if model.gated:
        hl=F.binary_cross_entropy_with_logits(hand,data.hand[ids,horizon,a],reduction="none")
        pl=F.binary_cross_entropy_with_logits(pair,data.pair[ids,horizon,a],reduction="none")
        loss=loss+.12*(hl*mask).sum()/mask.sum()+.12*(pl*valid).sum()/valid.sum().clamp_min(1)
    return loss


def fit(train, valid, seed, gated, out, epochs):
    torch.manual_seed(seed)
    device=train.state.device
    model=ContactTransfer(gated).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    best=np.inf
    indices=torch.arange(len(train.ids)*2,device=device)
    v=torch.arange(len(valid.ids)*2,device=device)
    for epoch in range(epochs):
        for batch in indices[torch.randperm(len(indices),device=device)].split(48):
            order=torch.randperm(len(train.actions),device=device)
            loss=objective(model,train,batch//2,batch%2,order)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
        if epoch+1 in (int(.68*epochs), int(.86*epochs)):
            for group in optimizer.param_groups:
                group["lr"] *= .45
        if (epoch+1)%4==0 or epoch+1==epochs:
            with torch.no_grad():
                score=objective(model,valid,v//2,v%2).item()
            if score < best:
                best=score
                torch.save(model.state_dict(),out)
    model.load_state_dict(torch.load(out,map_location=device,weights_only=True))
    return model.eval()
