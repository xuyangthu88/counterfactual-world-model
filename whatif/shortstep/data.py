"""Phase and window metadata, plus simulator labels used only by training and scoring."""
import numpy as np
import torch

from whatif.effects.data import load as coarse_load, pair_ids
from .collect import DATA, PROTOCOL


def schedule(d, domain):
    clock = PROTOCOL['cophy_clock' if domain=='cophy' else 'push_clock']
    ends = d['widths'][0].cumsum(0)
    times = ends.new_tensor(clock)
    parent = torch.searchsorted(ends, times)
    widths = torch.diff(times, prepend=times.new_zeros(1))
    take = torch.searchsorted(times,ends)
    if take[-1]>=len(times) or not torch.equal(ends,times[take]):
        raise ValueError('Every coarse endpoint must have a native label')
    d['end_idx'] = take.expand(len(d['x']),-1)
    d['window'] = (torch.arange(len(ends), device=parent.device)[:,None]==parent).expand(len(d['x']),-1,-1)
    d['native_widths'] = widths.expand(len(d['x']),-1)
    d['nominal_widths'] = d['widths'][:,parent]
    return d


def load(domain, split, device='cpu', *, labels=True):
    d = schedule(coarse_load(domain,split,device),domain)
    if not labels:
        return d
    with np.load(DATA/f'{domain}-{split}.npz') as file:
        for key in file.files:
            if key=='seed':
                if not np.array_equal(file[key],d[key].cpu().numpy()):
                    raise ValueError(f'Different layout order for {domain} {split}')
            else:
                d[key]=torch.as_tensor(file[key],device=device)
    d['native_off']=d['native_off'].float()
    return d


def paired_removal(d):
    b,k,n=d['a'].shape[:3]
    if 'removed_pair' in d:
        pair=d['removed_pair']
        target=d['native_off']
    else:
        i,j=pair_ids(n,d['x'].device)
        valid=d['role'].sum(-1).bool()
        valid=valid[:,i]&valid[:,j]
        # Sample the same editable pair for all branches in a factual situation.
        largest=d['native_off'].square().sum(-1).amax((1,3,4))
        pair=torch.multinomial(valid.float()*(1+4*(largest>.1**2)),1)[:,0]
        target=d['native_off'][torch.arange(b,device=pair.device),:,pair]
    return pair[:,None,None].expand(-1,k,-1),target
