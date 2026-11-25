"""One-macro horizon, shared Gaussian scenarios, and optional distributional decisions."""
import math
import torch

DRAWS, CANDIDATES, MOVES, CONTROL_N = 4, 24, 12, 64
KL_WEIGHT, HARM_MARGIN, HARM_LIMIT = .12, .04, .4
POLICIES = ['value','delta','delta_kl','pets','cophynet']


def select(model,policy,d,seed,goal_head=1,prior=None):
    if policy=='actor':
        return 0
    preference=0. if prior is None else .075*prior.clamp(min=-6.)
    if policy in ('pets','cophynet'):
        return (model(d)['r'][0,:,goal_head]+preference).argmax().item()
    k = d['a'].shape[1]
    d = {key:v.expand(DRAWS,*v.shape[1:]) for key,v in d.items()}
    gen = torch.Generator(device=d['x'].device).manual_seed(int(seed))
    half = torch.randn(DRAWS//2,d['x'].size(1),4,device=d['x'].device,generator=gen)
    d['eps'] = torch.cat([half,-half])
    out = model(d)
    incumbent = (out['r'][:,:,goal_head].mean(0)+preference).argmax()
    if policy=='value':
        return incumbent.item()
    i = torch.arange(k,device=d['x'].device)
    j = incumbent.expand_as(i)
    _,(mean,sd) = model.contrast(out,i,j)
    mean = mean[:,:,goal_head]
    sd = sd[:,:,goal_head]+.055
    cdf = torch.special.ndtr
    benefit = cdf(mean/sd).mean(0).clamp(1e-6,1-1e-6)
    harm = cdf((-HARM_MARGIN-mean)/sd).mean(0)
    divergence = math.log(2)+benefit*benefit.log()+(1-benefit)*(1-benefit).log()
    score = mean.mean(0)
    if prior is not None:
        score=score+preference-preference[incumbent]
    if policy=='delta_kl':
        score = score+KL_WEIGHT*torch.sign(benefit-.5)*divergence
    score = score.masked_fill(harm>HARM_LIMIT,-torch.inf)
    score[incumbent] = 0
    return score.argmax().item()
